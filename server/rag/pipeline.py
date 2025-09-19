import os, json
from dataclasses import dataclass
from typing import Any, Dict, List
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from datetime import datetime, timezone
<<<<<<< HEAD
import time
=======
>>>>>>> 4ce25ed9c119a7e9fed362c5919ea09db09f8ff0

from ingestion.models import MessageChunk
from utils.cleaning import sanitize_metadata
from logging_config import get_logger, log_rag_operation

load_dotenv()
logger = get_logger(__name__)

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
    score: float

class RAGPipelineGM:
    def __init__(self, batch_size: int = 64):
        logger.info(f"Initializing RAG pipeline with batch_size={batch_size}")

        # TODO: switch to remote chrome client in the future
        # self.client = chromadb.HttpClient(
        #     host=CHROMADB_SERVER,
        #     settings=Settings(
        #         chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        #         chroma_client_auth_credentials=CHROMADB_TOKEN,
        #     ),
        # )

        try:
            self.client = chromadb.PersistentClient(path="index/chroma")
            logger.info("Connected to ChromaDB persistent client")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {str(e)}")
            raise

        try:
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Connected to collection '{COLLECTION_NAME}'")
        except Exception as e:
            logger.error(f"Failed to get/create collection '{COLLECTION_NAME}': {str(e)}")
            raise

        try:
            self.llm = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("Connected to OpenAI API")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

        self.batch_size = batch_size
        logger.info("RAG pipeline initialization completed successfully")

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        logger.debug(f"Generating embeddings for {len(texts)} texts in batches of {self.batch_size}")
        start_time = time.time()

        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i+self.batch_size]
            logger.debug(f"Processing embedding batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")

            try:
                resp = self.llm.embeddings.create(model=EMBED_MODEL, input=chunk)
                out.extend([e.embedding for e in resp.data])
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch starting at index {i}: {str(e)}")
                raise

        processing_time = time.time() - start_time
        logger.info(f"Generated {len(texts)} embeddings in {processing_time:.3f}s")
        return out

    def add_messages(self, chunks: List[MessageChunk]):
        """Add message chunks to the vector database."""
        logger.info(f"Adding {len(chunks)} message chunks to collection")
        start_time = time.time()

        try:
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

            processing_time = time.time() - start_time
            logger.info(f"Successfully indexed {len(chunks)} message-chunks into collection '{COLLECTION_NAME}' in {processing_time:.3f}s")
            log_rag_operation(logger, "add_messages", results_count=len(chunks), processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to add messages after {processing_time:.3f}s: {str(e)}")
            raise

    def retrieve(self, query: str, k: int = 6) -> List[RetrievedHit]:
        """Retrieve relevant documents for a query."""
        logger.debug(f"Retrieving {k} documents for query: {query[:50]}...")
        start_time = time.time()

        try:
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
                sim = 1.0 - float(dist)
                hits.append(RetrievedHit(text=doc, meta=meta, score=sim))

            processing_time = time.time() - start_time
            logger.info(f"Retrieved {len(hits)} documents in {processing_time:.3f}s")
            log_rag_operation(logger, "retrieve", query, len(hits), processing_time)

            return hits

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to retrieve documents after {processing_time:.3f}s: {str(e)}")
            raise

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
            sender = h.meta.get("sender_name","Unknown")
            date = (h.meta.get("created_at_iso") or "")[:10]
            mtype = h.meta.get("msg_type","")
            ctx += f"[{i}] ({mtype}) ({sender} • {date})\n{h.text}\n\n"
        user = (
            f"{ctx}--- QUESTION ---\n{query}\n\n"
            "Write only the answer body (no headings, no inline citations)."
        )
        return [{"role":"system","content":sys}, {"role":"user","content":user}]

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
            if date_s:
                lines.append(f"[{i}] {sender} — {date_s}")
            else:
                lines.append(f"[{i}] {sender}")

        return "\n".join(lines)


    def generate(self, query: str, k: int = 6) -> str:
        """Generate a response using RAG pipeline."""
        logger.info(f"Generating response for query: {query[:100]}...")
        start_time = time.time()

        try:
            hits = self.retrieve(query, k=k)
            if not hits:
                logger.warning(f"No relevant documents found for query: {query[:50]}...")
                return "I couldn't find anything relevant in the chat."

            logger.debug(f"Building prompt with {len(hits)} retrieved documents")
            msgs = self._build_prompt(query, hits)

            logger.debug(f"Calling OpenAI API with model {CHAT_MODEL}")
            resp = self.llm.chat.completions.create(
                model=CHAT_MODEL,
                messages=msgs,
                temperature=0.3,
                max_tokens=350,
            )

            body = resp.choices[0].message.content.strip()
            citations = self._format_citations(hits)
            result = f"{body}\n\n{citations}"

            processing_time = time.time() - start_time
            logger.info(f"Generated response in {processing_time:.3f}s (length: {len(result)} chars)")
            log_rag_operation(logger, "generate", query, len(hits), processing_time)

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to generate response after {processing_time:.3f}s: {str(e)}")
            raise
