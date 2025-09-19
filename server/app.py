import re, time, asyncio
from typing import Dict, Any
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.pipeline import RAGPipelineGM
from logging_config import setup_logging_from_env, get_logger, log_request_info, log_bot_interaction
from config.settings import settings

setup_logging_from_env()
log = get_logger(__name__)

# Use settings instead of direct environment variable access
BOT_ID = settings.GROUPME_BOT_ID
BOT_NAME = settings.GROUPME_BOT_NAME
OPENAI_API_KEY = settings.OPENAI_API_KEY
POST_URL = f"{settings.GROUPME_API_URL}/bots/post"

MENTION_RE = re.compile(r"(^|\s)@eme([^\w]|$)", re.IGNORECASE)

log.info("Initializing RAG pipeline...")
rag = RAGPipelineGM()
log.info("RAG pipeline initialized successfully")

app = FastAPI(title="GroupMe Vector Bot", version="1.0.0")

# Debug: Log settings
log.info(f"Environment: {settings.ENV}")
log.info(f"CORS allowed origins: {settings.ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

log.info("FastAPI application initialized")

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

class ChatMessage(BaseModel):
    message: str

def is_mention_of_bot(msg: GroupMeMessage) -> bool:
    txt = (msg.text or "").strip()
    if not txt:
        return False

    if MENTION_RE.search(txt):
        return True

    for att in msg.attachments:
        if isinstance(att, dict) and att.get("type") == "mentions":
            return True
    return False

async def post_message(text: str):
    """Post a message to GroupMe via the bot API."""
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            data = {"bot_id": BOT_ID, "text": text}
            log.debug(f"Posting message to GroupMe: {text[:100]}...")
            r = await client.post(POST_URL, json=data)
            r.raise_for_status()
            response_time = time.time() - start_time
            log.info(f"Successfully posted message to GroupMe in {response_time:.3f}s")
    except Exception as e:
        response_time = time.time() - start_time
        log.error(f"Failed to post message to GroupMe after {response_time:.3f}s: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main landing page."""
    log.debug("Serving root page")
    html = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta
      name="viewport"
      content="width=device-width, initial-scale=1, viewport-fit=cover"
    />
    <title>eme</title>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400&display=swap');

      :root {
        --gray-00: #111110;
        --gray-06: #3b3a37;
        --gray-09: #6f6d66;
        --gray-11: #b5b3ad;
        --gray-A03: #e3e2de;
      }

      html, body {
        margin: 0;
        padding: 0;
        background: var(--gray-00);
        color: var(--gray-11);
        font-family: "JetBrains Mono", monospace;
      }

      .wrap {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 48px 16px;
      }

      .card {
        max-width: 600px;
        width: 100%;
        border: 1px solid var(--gray-06);
        padding: 40px 32px;
        background: var(--gray-00);
      }

      h1 {
        margin: 0 0 24px 0;
        font-size: 24px;
        font-weight: 400;
      }

      p {
        margin: 0 0 24px 0;
        font-size: 16px;
        line-height: 1.6;
      }

      a {
        color: var(--gray-11);
        text-decoration: underline;
      }

      .footer {
        margin-top: 32px;
        font-size: 14px;
        color: var(--gray-09);
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <main class="card">
        <h1>eme</h1>
        <p>
          hi! it seems that you found the server url for eme, good find! sadly, there's not much that you can do here...</code>
        </p>
        <p>
         to chat with eme, please go to the <a href="https://groupme.com/join_group/89417887/c1x6DI3U" target="_blank">emerging coders groupme</a> and tag @eme in a message. if you need help, please reach out to <a href="https://ethanpinedaa.dev/" target="_blank" rel="noopener noreferrer">ethan pineda</a>.
        </p>
        <div class="footer">
          built by <a href="https://ethanpinedaa.dev/" target="_blank" rel="noopener noreferrer">ethan pineda</a> ·
          for <a href="https://emergingcoders.org/" target="_blank" rel="noopener noreferrer">emerging coders</a>
        </div>
      </main>
    </div>
  </body>
</html>"""
    return HTMLResponse(html)



@app.post("/bot/callback")
async def bot_callback(req: Request):
    """Handle GroupMe bot callback messages."""
    start_time = time.time()
    client_ip = req.client.host if req.client else "unknown"
    user_agent = req.headers.get("user-agent", "unknown")
    payload = await req.json()
    msg = GroupMeMessage(**payload)
    if msg.sender_type != "user":
        return Response(status_code=204)

    if not is_mention_of_bot(msg):
        return Response(status_code=204)

    query = re.sub(MENTION_RE, " ", (msg.text or "")).strip()
    if not query:
        await post_message("Hi! Ask me something like “@eme should I take CS214 and CS211 at the same time?”")
        return Response(status_code=200)

    log.info(f"@{BOT_NAME} asked: {query}")

    try:
        payload = await req.json()
        log.debug(f"Received bot callback from {client_ip}: {payload}")

        msg = GroupMeMessage(**payload)

        log_request_info(log, "POST", "/bot/callback", 200, time.time() - start_time, user_agent, client_ip)

        if msg.sender_type != "user":
            log.debug(f"Skipping non-user message from sender_type: {msg.sender_type}")
            return Response(status_code=204)

        if not is_mention_of_bot(msg):
            log.debug(f"Bot not mentioned in message from {msg.sender_id}")
            return Response(status_code=204)

        query = re.sub(MENTION_RE, " ", (msg.text or "")).strip()
        if not query:
            log.info(f"Empty query from user {msg.sender_id}, sending help message")
            await post_message("Hi! Ask me something like \"@eme should I take CS214 and CS211 at the same time?\"")
            return Response(status_code=200)

        log.info(f"Processing query from user {msg.sender_id}: {query[:100]}...")

        rag_start_time = time.time()
        try:
            answer = rag.generate(query, k=6)
            rag_time = time.time() - rag_start_time

            if len(answer) > 950:
                answer = answer[:950] + "…"
                log.warning(f"Truncated response for user {msg.sender_id} (was {len(answer)} chars)")

            log_bot_interaction(log, query, answer, rag_time, msg.sender_id, msg.group_id)

            await post_message(answer)
            log.info(f"Successfully responded to user {msg.sender_id} in {rag_time:.3f}s")

        except Exception as e:
            rag_time = time.time() - rag_start_time
            log.error(f"RAG generation failed for user {msg.sender_id} after {rag_time:.3f}s: {str(e)}")
            await post_message("Sorry, I hit an error answering that.")
            return Response(status_code=200)

        return Response(status_code=200)

    except Exception as e:
        response_time = time.time() - start_time
        log.error(f"Bot callback error after {response_time:.3f}s: {str(e)}")
        log_request_info(log, "POST", "/bot/callback", 500, response_time, user_agent, client_ip)
        return Response(status_code=500)

@app.post("/chat")
async def chat_endpoint(chat_msg: ChatMessage):
    """Chat endpoint for frontend."""
    start_time = time.time()
    query = chat_msg.message.strip()

    if not query:
        return StreamingResponse(
            iter(["Please ask me something!"],),
            media_type="text/plain"
        )

    log.info(f"Frontend chat query: {query[:100]}...")

    try:
        rag_start_time = time.time()
        answer = rag.generate(query, k=6)
        rag_time = time.time() - rag_start_time

        # No truncation needed for frontend - it can handle longer responses
        log.info(f"Successfully generated response for frontend in {rag_time:.3f}s")

        # Stream the response
        async def generate_response():
            for char in answer:
                yield char.encode('utf-8')
                await asyncio.sleep(0.01)

        return StreamingResponse(
            generate_response(),
            media_type="text/plain"
        )

    except Exception as e:
        rag_time = time.time() - rag_start_time
        log.error(f"RAG generation failed for frontend after {rag_time:.3f}s: {str(e)}")
        return StreamingResponse(
            iter(["Sorry, I hit an error answering that."]),
            media_type="text/plain"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    log.debug("Health check requested")
    return {"status": "ok"}

@app.get("/cors-test")
async def cors_test(request: Request):
    """CORS test endpoint to debug CORS issues."""
    origin = request.headers.get("origin")
    log.info(f"CORS test request from origin: {origin}")
    return {
        "status": "ok",
        "origin": origin,
        "allowed_origins": settings.ALLOWED_ORIGINS,
        "env": settings.ENV
    }
