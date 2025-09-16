import os, re, time, logging
from typing import Dict, Any
import httpx
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

from rag.pipeline import RAGPipelineGM  # your pipeline that calls Chroma + LLM

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("eme")

BOT_ID = os.environ.get("GROUPME_BOT_ID")          # required to post
BOT_NAME = os.environ.get("GROUPME_BOT_NAME", "eme")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # used by your pipeline
POST_URL = "https://api.groupme.com/v3/bots/post"

# Simple mention detector:
MENTION_RE = re.compile(r"(^|\s)@eme([^\w]|$)", re.IGNORECASE)

# Initialize pipeline once (uses your PersistentClient by default)
rag = RAGPipelineGM()

app = FastAPI()

class GroupMeMessage(BaseModel):
    attachments: list = []
    avatar_url: str | None = None
    created_at: int
    group_id: str
    id: str
    name: str
    sender_id: str
    sender_type: str
    source_guid: str | None = None
    system: bool
    text: str | None = None
    user_id: str | None = None

def is_mention_of_bot(msg: GroupMeMessage) -> bool:
    txt = (msg.text or "").strip()
    if not txt:
        return False

    # 1) Text contains @eme
    if MENTION_RE.search(txt):
        return True

    # 2) Or an explicit mention attachment (GroupMe sometimes includes this)
    for att in msg.attachments:
        if isinstance(att, dict) and att.get("type") == "mentions":
            # mentions attachment has loci / user_ids; we don't have bot user id,
            # so fallback to text match. You can expand this if you capture it.
            return True
    return False

async def post_message(text: str):
    async with httpx.AsyncClient(timeout=10) as client:
        data = {"bot_id": BOT_ID, "text": text}
        r = await client.post(POST_URL, json=data)
        r.raise_for_status()

@app.post("/bot/callback")
async def bot_callback(req: Request):
    # GroupMe POSTs a single message payload
    payload = await req.json()
    msg = GroupMeMessage(**payload)

    # Avoid loops: ignore our own bot & system messages
    if msg.sender_type != "user":
        return Response(status_code=204)

    # Only respond when mentioned
    if not is_mention_of_bot(msg):
        return Response(status_code=204)

    # Build a query by stripping the mention
    query = re.sub(MENTION_RE, " ", (msg.text or "")).strip()
    if not query:
        await post_message("Hi! Ask me something like “@eme when is Space Apps?”")
        return Response(status_code=200)

    log.info(f"@{BOT_NAME} asked: {query}")

    # Generate answer from your RAG index
    try:
        answer = rag.generate(query, k=6)
        # Keep replies short-ish; GroupMe messages cap at ~1000 chars
        if len(answer) > 950:
            answer = answer[:950] + "…"
        await post_message(answer)
    except Exception as e:
        log.exception("Failed to answer")
        await post_message("Sorry, I hit an error answering that.")

    return Response(status_code=200)
