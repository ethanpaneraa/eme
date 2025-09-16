import os, re, time, logging
from typing import Dict, Any
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag.pipeline import RAGPipelineGM

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("eme")

BOT_ID = os.environ.get("GROUPME_BOT_ID")
BOT_NAME = os.environ.get("GROUPME_BOT_NAME", "eme")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
POST_URL = os.environ.get("GROUPME_API_URL" + "/bots/post", "https://api.groupme.com/v3") + "/bots/post"

MENTION_RE = re.compile(r"(^|\s)@eme([^\w]|$)", re.IGNORECASE)

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

    if MENTION_RE.search(txt):
        return True

    for att in msg.attachments:
        if isinstance(att, dict) and att.get("type") == "mentions":
            return True
    return False

async def post_message(text: str):
    async with httpx.AsyncClient(timeout=10) as client:
        data = {"bot_id": BOT_ID, "text": text}
        r = await client.post(POST_URL, json=data)
        r.raise_for_status()

@app.get("/", response_class=HTMLResponse)
async def root():
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
      @font-face {
        font-family: 'Geist Mono';
        font-style: normal;
        font-weight: 400;
        font-display: swap;
        src: url('https://vercel.com/font/geist-mono/Geist-Mono-Regular.woff2') format('woff2');
      }

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
        font-family: "Geist Mono", monospace;
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
          project for <a href="https://emergingcoders.org/" target="_blank" rel="noopener noreferrer">emerging coders</a>
        </div>
      </main>
    </div>
  </body>
</html>"""
    return HTMLResponse(html)



@app.post("/bot/callback")
async def bot_callback(req: Request):
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
        answer = rag.generate(query, k=6)
        if len(answer) > 950:
            answer = answer[:950] + "…"
        await post_message(answer)
    except Exception as e:
        log.exception("Failed to answer")
        await post_message("Sorry, I hit an error answering that.")

    return Response(status_code=200)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
