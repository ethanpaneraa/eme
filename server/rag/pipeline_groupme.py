import os, json, re
from dataclasses import dataclass
from typing import Any, Dict, List
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from datetime import datetime, timezone
import time

from ingestion.models import MessageChunk
from utils.cleaning import sanitize_metadata
from logging_config import get_logger, log_rag_operation

# ============================================================================
# PAPERNU INTEGRATION: Import course data models
# ============================================================================
from ingestion.models import FullCourseRecord
from ingestion.papernu_loader import make_context_from_papernu_data

load_dotenv()
logger = get_logger(__name__)

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

@dataclass
class RetrievedHit:
    text: str
    meta: Dict[str, Any]
    score: float

class RAGPipelineGM:
    def __init__(self, batch_size: int = 64):
        logger.info(f"Initializing RAG pipeline with batch_size={batch_size}")
        # figure out which kind of vector database to use
        self.backend = (VECTOR_BACKEND or ("pinecone" if PINECONE_API_KEY else "chroma")).lower()
        logger.info(f"Vector backend selected: {self.backend}")

        # ============================================================================
        # PAPERNU INTEGRATION: Load course catalog data
        # ============================================================================
        try:
            self.course_records: List[FullCourseRecord] = make_context_from_papernu_data()
            logger.info(f"Loaded {len(self.course_records)} course records from PaperNU data")
        except FileNotFoundError as e:
            logger.error(f"PaperNU data file not found: {str(e)}")
            self.course_records = []
            logger.warning("Continuing without PaperNU course data - file missing")
        except Exception as e:
            logger.error(f"Failed to load PaperNU course records: {str(e)}")
            self.course_records = []
            logger.warning("Continuing without PaperNU course data")

        if self.backend == "pinecone":
            try:
                if not PINECONE_API_KEY:
                    raise ValueError("PINECONE_API_KEY environment variable is required for Pinecone backend")
                from pinecone import Pinecone
                import pinecone as _pinecone_mod

                self._pinecone_mod = _pinecone_mod
                self.pc = Pinecone(api_key=PINECONE_API_KEY)
                logger.info("Connected to Pinecone")

                try:
                    indexes = self.pc.list_indexes()
                    index_names = [idx.name for idx in indexes]
                    if PINECONE_INDEX_NAME not in index_names:
                        logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' (dim=1536, metric=cosine)")
                        self.pc.create_index(
                            name=PINECONE_INDEX_NAME,
                            dimension=1536,
                            metric="cosine",
                            spec=self._pinecone_mod.ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
                        )
                        logger.info(f"Created Pinecone index '{PINECONE_INDEX_NAME}'")
                    self.index = self.pc.Index(PINECONE_INDEX_NAME)
                    logger.info(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'")
                except Exception as e:
                    logger.error(f"Failed to get/create Pinecone index: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone backend: {str(e)}")
                raise
        else:
            # Default to ChromaDB persistent client
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
                
                # ============================================================================
                # PAPERNU INTEGRATION: Create separate collection for course data
                # ============================================================================
                self.papernu_collection = self.client.get_or_create_collection(
                    name=f"{COLLECTION_NAME}_papernu",
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(f"Connected to PaperNU collection '{COLLECTION_NAME}_papernu'")
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

            if self.backend == "pinecone":
                vectors = []
                for id_, emb, meta, text in zip(ids, embs, metas, texts):
                    meta = dict(meta)
                    meta["text"] = text
                    vectors.append({"id": id_, "values": emb, "metadata": meta})

                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i+batch_size]
                    self.index.upsert(vectors=batch)
                    logger.debug(f"Pinecone upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            else:
                # Insert into ChromaDB
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

    # ============================================================================
    # PAPERNU INTEGRATION: Add course records to vector database
    # ============================================================================
    def add_course_records(self, records: List[FullCourseRecord]):
        """Add PaperNU course records to the vector database."""
        logger.info(f"Adding {len(records)} course records to PaperNU collection")
        start_time = time.time()

        try:
            ids = [f"{r.subject} {r.catalog_number}" for r in records]
            texts = [r.get_message() for r in records]
            embeddings = self._batch_embed(texts)

            if self.backend == "pinecone":
                # For Pinecone, we'll use a namespace to separate course data
                vectors = []
                for id_, emb, text in zip(ids, embeddings, texts):
                    meta = {"text": text, "source": "papernu", "type": "course"}
                    vectors.append({"id": id_, "values": emb, "metadata": meta})

                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i+batch_size]
                    self.index.upsert(vectors=batch, namespace="papernu")
                    logger.debug(f"Pinecone upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            else:
                # Insert into ChromaDB PaperNU collection
                self.papernu_collection.add(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                )

            processing_time = time.time() - start_time
            logger.info(f"Successfully indexed {len(records)} course records in {processing_time:.3f}s")
            log_rag_operation(logger, "add_course_records", results_count=len(records), processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to add course records after {processing_time:.3f}s: {str(e)}")
            raise

    def retrieve(self, query: str, k: int = 6) -> List[RetrievedHit]:
        """Retrieve relevant documents for a query."""
        logger.debug(f"Retrieving {k} documents for query: {query[:50]}...")
        start_time = time.time()

        try:
            q_emb = self._batch_embed([query])[0]
            hits = []

            if self.backend == "pinecone":
                result = self.index.query(vector=q_emb, top_k=k, include_metadata=True)
                for match in result.matches:
                    meta = dict(match.metadata or {})
                    text = meta.pop("text", "")
                    hits.append(RetrievedHit(text=text, meta=meta, score=float(match.score)))
            else:
                res = self.collection.query(
                    query_embeddings=[q_emb],
                    n_results=k,
                    include=["documents", "metadatas", "distances"],
                )

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

    # ============================================================================
    # PAPERNU INTEGRATION: Dual retrieval from both GroupMe and PaperNU sources
    # ============================================================================
    def retrieve_papernu(self, query: str, k: int = 6) -> List[RetrievedHit]:
        """Retrieve relevant course information from PaperNU collection."""
        logger.debug(f"Retrieving {k} course documents for query: {query[:50]}...")
        start_time = time.time()

        try:
            # Enhance query with catalog number matches
            matched_courses: str = ",".join(self._match_catalog_number_to_course(query))
            query = query + "\n" + matched_courses

            embedded_query = self._batch_embed([query])[0]
            hits = []

            if self.backend == "pinecone":
                result = self.index.query(
                    vector=embedded_query,
                    top_k=k, 
                    include_metadata=True,
                    namespace="papernu"
                )
                for match in result.matches:
                    meta = dict(match.metadata or {})
                    text = meta.pop("text", "")
                    meta["source"] = "papernu"
                    hits.append(RetrievedHit(text=text, meta=meta, score=float(match.score)))
            else:
                res = self.papernu_collection.query(
                    query_embeddings=[embedded_query],
                    n_results=k,
                    include=["documents", "metadatas", "distances"],
                )

                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]

                for doc, meta, dist in zip(docs, metas, dists):
                    sim = 1.0 - float(dist)
                    if meta is None:
                        meta = {}
                    meta["source"] = "papernu"
                    hits.append(RetrievedHit(text=doc, meta=meta, score=sim))

            processing_time = time.time() - start_time
            logger.info(f"Retrieved {len(hits)} PaperNU documents in {processing_time:.3f}s")

            return hits

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to retrieve PaperNU documents after {processing_time:.3f}s: {str(e)}")
            raise

    def retrieve_combined(self, query: str, k_groupme: int = 6, k_papernu: int = 6) -> tuple[List[RetrievedHit], List[RetrievedHit]]:
        """
        Retrieve from both GroupMe messages and PaperNU course data.
        Returns tuple of (groupme_hits, papernu_hits).
        """
        logger.info(f"Combined retrieval: {k_groupme} GroupMe + {k_papernu} PaperNU docs")
        
        try:
            groupme_hits = self.retrieve(query, k=k_groupme)
            papernu_hits = self.retrieve_papernu(query, k=k_papernu)
            
            logger.info(f"Combined retrieval complete: {len(groupme_hits)} GroupMe + {len(papernu_hits)} PaperNU")
            return (groupme_hits, papernu_hits)
        except Exception as e:
            logger.error(f"Failed combined retrieval: {str(e)}")
            raise

    # ============================================================================
    # PAPERNU INTEGRATION: Course catalog number matching
    # ============================================================================
    def _match_catalog_number_to_course(self, query: str) -> List[str]:
        """
        Given a query containing catalog numbers (e.g., "CS336", "348"), match them to course names.
        
        This improves RAG matching because queries like "what are the prereqs for CS330?" 
        don't perform well matching with just the number. By expanding to include
        "CS330: Human Computer Interaction", we greatly improve embedding similarity.
        
        Returns list of formatted course strings like "336: Design & Analysis of Algorithms. Description..."
        """
        def _find_catalog_number(query: str) -> List[str]:
            """Extract 3-digit course numbers (with optional dash) from query."""
            COURSE_NUMBER_LEN = 3
            nums_dash = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-']
            ret = []

            for i in range(len(query) - COURSE_NUMBER_LEN + 1): 
                segment: str = query[i:i + COURSE_NUMBER_LEN]
                if all(c in nums_dash for c in segment):
                    ret.append(segment)
            return ret
        
        potential_catalog_numbers: List[str] = _find_catalog_number(query)
        names: List[str] = []

        for catalog_number in potential_catalog_numbers:
            for record in self.course_records:
                if catalog_number in record.catalog_number:
                    names.append(f"{catalog_number}: {record.name}. {record.description}")
        
        logger.debug(f"Matched catalog numbers in query: {potential_catalog_numbers} -> {len(names)} courses")
        return names

    def _is_general_question(self, query: str) -> bool:
        """Detect if this is a general question about the bot that doesn't need GroupMe context."""
        general_patterns = [
            r'\bwhat\s+are\s+you\b',
            r'\bwho\s+are\s+you\b',
            r'\bwhat\s+is\s+your\s+name\b',
            r'\bwhat\s+do\s+you\s+do\b',
            r'\bhow\s+do\s+you\s+work\b',
            r'\bwhat\s+can\s+you\s+do\b',
            r'\bwhat\s+are\s+your\s+capabilities\b',
            r'\bhelp\b',
            r'\bcommands?\b',
            r'\bhow\s+to\s+use\b',
            r'\bwhat\s+should\s+I\s+ask\b'
        ]

        query_lower = query.lower().strip()
        for pattern in general_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def _build_general_prompt(self, query: str) -> List[Dict[str, str]]:
        """Build prompt for general questions about the bot."""
        sys = (
            "You are eme, a helpful AI assistant for the Emerging Coders GroupMe chat. "
            "You can answer questions about courses, internships, career advice, and general college life "
            "based on the chat history. You're friendly, knowledgeable, and always try to be helpful. "
            "Keep responses conversational and concise (under 200 words)."
        )

        user = (
            f"Question: {query}\n\n"
            "Please answer this question about yourself or your capabilities. "
            "Be helpful and friendly in your response."
        )

        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    def _build_prompt(self, query: str, hits: List[RetrievedHit]) -> List[Dict[str, str]]:
        sys = (
            "You answer newcomer questions using ONLY the provided GroupMe excerpts.\n"
            "Write a helpful, clear answer of roughly 120–200 words. Be specific and actionable.\n"
            "If the context doesn't contain the answer, say so briefly and suggest what to ask next.\n"
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

    # ============================================================================
    # PAPERNU INTEGRATION: Build hybrid prompt with both GroupMe and course data
    # ============================================================================
    def _build_hybrid_prompt(self, query: str, groupme_hits: List[RetrievedHit], papernu_hits: List[RetrievedHit]) -> List[Dict[str, str]]:
        """
        Build a prompt that includes both GroupMe chat excerpts and PaperNU course information.
        This allows the bot to answer questions using both community discussions and official course data.
        """
        sys = (
            "You are eme, a helpful AI assistant for the Emerging Coders community at Northwestern.\n"
            "You answer questions using two sources:\n"
            "1. GroupMe chat excerpts - community discussions, experiences, and advice\n"
            "2. Official course catalog data from paper.nu - course descriptions, prerequisites, etc.\n\n"
            "Write a helpful, clear answer of roughly 120–200 words. Be specific and actionable.\n"
            "When referencing course information, be authoritative. When referencing chat discussions, acknowledge it's community perspective.\n"
            "Do NOT include inline citations in the body."
        )
        
        ctx = ""
        
        # Add PaperNU course data first (official information)
        if papernu_hits:
            ctx += "\n--- OFFICIAL COURSE INFORMATION (from paper.nu) ---\n"
            for i, h in enumerate(papernu_hits, 1):
                ctx += f"[COURSE-{i}]\n{h.text}\n\n"
        
        # Add GroupMe excerpts (community discussions)
        if groupme_hits:
            ctx += "\n--- COMMUNITY DISCUSSIONS (from GroupMe) ---\n"
            for i, h in enumerate(groupme_hits, 1):
                sender = h.meta.get("sender_name","Unknown")
                date = (h.meta.get("created_at_iso") or "")[:10]
                mtype = h.meta.get("msg_type","")
                ctx += f"[CHAT-{i}] ({mtype}) ({sender} • {date})\n{h.text}\n\n"
        
        user = (
            f"{ctx}--- QUESTION ---\n{query}\n\n"
            "Provide a comprehensive answer using both official course information and community insights where relevant."
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
            # Check if this is a general question about the bot
            if self._is_general_question(query):
                logger.info(f"Detected general question, using general prompt: {query[:50]}...")
                msgs = self._build_general_prompt(query)

                logger.debug(f"Calling OpenAI API with model {CHAT_MODEL} for general question")
                resp = self.llm.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=msgs,
                    temperature=0.3,
                    max_tokens=350,
                )

                result = resp.choices[0].message.content.strip()
                processing_time = time.time() - start_time
                logger.info(f"Generated general response in {processing_time:.3f}s (length: {len(result)} chars)")
                log_rag_operation(logger, "generate_general", query, 0, processing_time)
                return result

            # ============================================================================
            # PAPERNU INTEGRATION: Use combined retrieval from both sources
            # ============================================================================
            # Retrieve from both GroupMe and PaperNU collections
            groupme_hits, papernu_hits = self.retrieve_combined(query, k_groupme=k, k_papernu=k)
            
            if not groupme_hits and not papernu_hits:
                logger.warning(f"No relevant documents found for query: {query[:50]}...")
                return "I couldn't find anything relevant in the chat or course catalog."

            # Use hybrid prompt if we have both sources, otherwise use appropriate single-source prompt
            if papernu_hits:
                logger.debug(f"Building hybrid prompt with {len(groupme_hits)} GroupMe + {len(papernu_hits)} PaperNU docs")
                msgs = self._build_hybrid_prompt(query, groupme_hits, papernu_hits)
            else:
                logger.debug(f"Building prompt with {len(groupme_hits)} GroupMe documents only")
                msgs = self._build_prompt(query, groupme_hits)

            logger.debug(f"Calling OpenAI API with model {CHAT_MODEL}")
            resp = self.llm.chat.completions.create(
                model=CHAT_MODEL,
                messages=msgs,
                temperature=0.3,
                max_tokens=350,
            )

            body = resp.choices[0].message.content.strip()
            # citations = self._format_citations(hits)
            #  result = f"{body}\n\n{citations}"
            result = body

            processing_time = time.time() - start_time
            total_hits = len(groupme_hits) + len(papernu_hits)
            logger.info(f"Generated hybrid response in {processing_time:.3f}s (length: {len(result)} chars, {len(groupme_hits)} GM + {len(papernu_hits)} PN hits)")
            log_rag_operation(logger, "generate_hybrid", query, total_hits, processing_time)

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to generate response after {processing_time:.3f}s: {str(e)}")
            raise

    def delete_all(self):
        """Delete all vectors from the index (only implemented for Pinecone backend)."""
        if self.backend != "pinecone":
            raise NotImplementedError("delete_all is only available for Pinecone backend")
        logger.info("Deleting all vectors from Pinecone index")
        try:
            self.index.delete(delete_all=True)
            logger.info("Deleted all vectors from Pinecone index")
        except Exception as e:
            # Some Pinecone deployments return 404 if no namespace exists yet
            if "Namespace not found" in str(e):
                logger.info("No existing namespace found; index already empty")
                return
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics (only implemented for Pinecone backend)."""
        if self.backend != "pinecone":
            return {"backend": "chroma"}
        try:
            stats = self.index.describe_index_stats()
            return {
                "backend": "pinecone",
                "total_vector_count": stats.total_vector_count,
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone index stats: {str(e)}")
            return {"backend": "pinecone", "error": str(e)}
