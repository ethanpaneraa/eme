# Pinecone Migration Guide

This document describes the migration from ChromaDB to Pinecone for the GroupMe vector database.

## Overview

The system has been migrated from using a local ChromaDB instance to Pinecone, a cloud-based vector database service. This provides better scalability, reliability, and performance for the RAG (Retrieval-Augmented Generation) pipeline.

## Environment Variables Required

Add these environment variables to your `.env` file:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=groupme-messages
PINECONE_ENVIRONMENT=us-east-1

# GroupMe Bot Configuration
GROUPME_BOT_ID=your_bot_id_here
GROUPME_BOT_NAME=eme
GROUPME_API_URL=https://api.groupme.com/v3

# Application Configuration
ENV=dev
FRONTEND_URL=http://localhost:5173
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd server
pip install -r requirements.txt
```

### 2. Set up Pinecone

1. Create a Pinecone account at https://www.pinecone.io/
2. Create a new project and get your API key
3. Add the API key to your `.env` file as `PINECONE_API_KEY`

### 3. Build the Pinecone Index

To ingest your data into Pinecone, run:

```bash
cd server
python scripts/build_pinecone_index.py --input_glob "data/raw/*.jsonl" --window 1 --clear
```

Options:

- `--input_glob`: Pattern for input JSONL files (default: "data/raw/\*.jsonl")
- `--window`: Context window size (default: 1)
- `--clear`: Clear existing index before adding new data

### 4. Test the Migration

Test the interactive chat:

```bash
cd server
python scripts/chat.py
```

Start the web server:

```bash
cd server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Key Changes Made

### New Files Created

- `rag/pinecone_pipeline.py`: New Pinecone-based RAG pipeline
- `scripts/build_pinecone_index.py`: Script to build Pinecone index

### Files Modified

- `requirements.txt`: Added `pinecone-client` dependency
- `config/settings.py`: Added Pinecone configuration variables
- `app.py`: Updated to use `PineconeRAGPipeline` instead of `RAGPipelineGM`
- `scripts/chat.py`: Updated to use `PineconeRAGPipeline`

### Legacy Files (Still Available)

- `rag/pipeline.py`: Original ChromaDB implementation (kept for reference)
- `scripts/build_index.py`: Original ChromaDB indexing script

## Features

### Pinecone Pipeline Features

- **Automatic Index Creation**: Creates index if it doesn't exist
- **Batch Processing**: Efficient batch upserts for large datasets
- **Index Management**: Methods to clear index and get statistics
- **Same Interface**: Drop-in replacement for ChromaDB pipeline

### Index Configuration

- **Dimension**: 1536 (for OpenAI text-embedding-3-small)
- **Metric**: Cosine similarity
- **Cloud**: AWS (configurable via PINECONE_ENVIRONMENT)

## Data Structure

The system processes GroupMe messages into three types of chunks:

1. **Q&A Chunks**: Questions with their replies and nearby answers
2. **Announcement Chunks**: Important announcements, job postings, etc.
3. **Context Chunks**: Regular messages with surrounding context

Each chunk includes metadata:

- `msg_id`: Original message ID
- `group_id`: GroupMe group ID
- `sender_id`: Message sender ID
- `sender_name`: Sender's display name
- `created_at_iso`: ISO timestamp
- `window`: Context window size
- `raw_len`: Original message length
- `msg_type`: Type of chunk (qna, announcement, context, system)

## Migration Benefits

1. **Scalability**: Pinecone handles millions of vectors efficiently
2. **Reliability**: Cloud-hosted with built-in redundancy
3. **Performance**: Optimized for vector similarity search
4. **Maintenance**: No need to manage local database files
5. **Features**: Advanced filtering, namespaces, and analytics

## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `PINECONE_API_KEY` is set in your `.env` file
2. **Index Creation Failed**: Check your Pinecone account limits and region settings
3. **Embedding Dimension Mismatch**: Ensure you're using `text-embedding-3-small` (1536 dimensions)

### Logs

Check the application logs for detailed error messages:

- `logs/app.log`: General application logs
- `logs/error.log`: Error-specific logs

### Index Statistics

You can check index statistics in the build script output or by calling:

```python
from rag.pinecone_pipeline import PineconeRAGPipeline
rag = PineconeRAGPipeline()
stats = rag.get_stats()
print(stats)
```

## Next Steps

1. Set up your Pinecone account and API key
2. Run the migration script to index your data
3. Test the system with sample queries
4. Deploy to production environment

The migration maintains full compatibility with the existing API and chat interface, so no changes are needed to the frontend or bot integration.
