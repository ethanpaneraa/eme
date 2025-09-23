### eme

RAG-powered assistant for the Emerging Coders GroupMe. The backend ingests GroupMe messages into a vector store (ChromaDB by default, optional Pinecone) and serves answers via FastAPI. A React + Vite frontend provides a simple chat UI that streams responses.

## Monorepo Layout

- `server/`: FastAPI app, RAG pipeline, ingestion and indexing scripts
- `frontend/`: React + Vite chat interface
- `data/`: Example project data (not used by the pipeline)
- `fetch_groupme_messages.sh`: Script to export full GroupMe message history to JSONL

## Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- curl and jq (for the GroupMe export script)
- OpenAI API key
- Optional: Pinecone account and API key

## Backend Setup (FastAPI + RAG)

1. Create a virtual environment and install dependencies. We recommend using `uv` (a fast Python package manager) with a project-local virtual environment:

```bash
cd server
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you need to install `uv` first:

```bash
# macOS (Homebrew)
brew install uv

# or via the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternative using Python + pip:

```bash
cd server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create `server/.env` with the needed variables:

```bash
# Runtime
ENV=dev
OPENAI_API_KEY=sk-...

# GroupMe bot (optional unless you are wiring up callbacks)
GROUPME_BOT_ID=your_bot_id
GROUPME_BOT_NAME=eme
GROUPME_API_URL=https://api.groupme.com/v3

# CORS/frontend
FRONTEND_URL=http://localhost:5173

# Vector backend selection: "chroma" (default) or "pinecone"
VECTOR_BACKEND=chroma

# Chroma (defaults are fine for local persistent client)
COLLECTION_NAME=groupme

# Pinecone (only if using Pinecone)
PINECONE_API_KEY=
PINECONE_INDEX_NAME=groupme-messages
PINECONE_ENVIRONMENT=us-east-1
```

3. Start the backend:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- `GET /health` — health check
- `POST /chat` — body `{ "message": "..." }`, streams back the answer
- `POST /bot/callback` — GroupMe bot webhook (requires bot setup on GroupMe)
- `GET /cors-test` — debug CORS headers

Quick test:

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What can you help with?"}'
```

## Exporting GroupMe Messages (JSONL)

Use the provided script to pull the entire message history for a group. You will need a GroupMe access token and the group ID.

```bash
# From repo root
chmod +x fetch_groupme_messages.sh
./fetch_groupme_messages.sh <GROUP_ID> <ACCESS_TOKEN> server/data/raw/messages.jsonl
```

The output file contains one JSON object per line.

## Building the Vector Index

Once you have one or more `.jsonl` dumps under `server/data/raw/`, build the index:

```bash
cd server
python scripts/build_index.py --input_glob "data/raw/*.jsonl" --window 1
```

Notes:

- Default backend is ChromaDB persistent client under `server/index/chroma`.
- To use Pinecone, set `VECTOR_BACKEND=pinecone` and provide `PINECONE_*` variables in `.env`.

## Frontend Setup (React + Vite)

1. Install and configure:

```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env
```

2. Start the dev server:

```bash
npm run dev
```

By default the frontend runs at `http://localhost:5173`. The backend CORS policy allows localhost dev origins and any `FRONTEND_URL` you set in the backend `.env`.

## How It Works (High-Level)

1. Ingestion: `fetch_groupme_messages.sh` exports raw messages to JSONL.
2. Processing: `server/ingestion/loader.py` normalizes, threads, and creates chunks (Q&A, announcements, and context windows).
3. Indexing: `server/rag/pipeline.py` batches embeddings via OpenAI and stores vectors in ChromaDB or Pinecone.
4. Retrieval + Generation: `/chat` retrieves top-k chunks and generates an answer via OpenAI chat completions.

## Configuration Reference

Backend environment variables (see `server/config/settings.py` and `server/rag/pipeline.py`):

- `ENV`: `dev` or `prod` (controls CORS defaults)
- `OPENAI_API_KEY`: required
- `GROUPME_BOT_ID`, `GROUPME_BOT_NAME`, `GROUPME_API_URL`: for GroupMe bot posting and callbacks
- `FRONTEND_URL`: appended to allowed CORS origins in non-dev
- `VECTOR_BACKEND`: `chroma` (default) or `pinecone`
- Chroma: `COLLECTION_NAME` (default `groupme`)
- Pinecone: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_ENVIRONMENT`

Frontend environment variables:

- `VITE_API_URL`: base URL of the backend (e.g., `http://localhost:8000`)

## Troubleshooting

- No responses or 500s: verify `OPENAI_API_KEY` is set and valid.
- CORS errors in browser: ensure backend `ENV=dev` or set `FRONTEND_URL` to match the React dev URL.
- Empty retrievals: confirm `server/data/raw/*.jsonl` exists and rerun the index build.
- Pinecone errors: double-check region and index name; omit Pinecone vars to fall back to Chroma.

## Security and Data

Keep exported message data private. Do not commit `.env` files or raw exports. Use separate OpenAI and Pinecone keys for development and production.

\*\*If you need access to the production `.env` variables, reach out to [Ethan](mailto:joshuapineda66@gmail.com)

TODO:

## Scheduled incremental fetch (daily at midnight)

To securely fetch only new GroupMe messages and append to `server/data/raw/messages.jsonl`:

1. Create a `.env` at the repo root (not committed) with:

```bash
GROUPME_ACCESS_TOKEN=your_token_here
GROUPME_GROUP_ID=89417887
```

2. Test a one-off run:

```bash
bash server/scripts/fetch_incremental.sh
```

3. macOS launchd schedule (runs nightly at 00:00):

```bash
mkdir -p ~/Library/LaunchAgents
cp server/scripts/com.ec.groupme.fetch.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.ec.groupme.fetch.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.ec.groupme.fetch.plist
launchctl list | grep com.ec.groupme.fetch || true
```

Logs:

- Append log: `server/logs/fetch.log`
- Out/err: `server/logs/fetch.out`, `server/logs/fetch.err`

The script is incremental using `since_id` and appends oldest-first to preserve chronological order and avoid duplicates.
