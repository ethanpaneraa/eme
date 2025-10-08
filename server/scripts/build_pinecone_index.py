import argparse, pathlib, glob, sys

# Ensure project root (server/) is on sys.path when running from scripts/
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from server.ingestion.groupme_loader import load_jsonl, make_chunks_from_records
from rag.pipeline import RAGPipelineGM
from logging_config import setup_logging_from_env, get_logger

# Initialize logging
setup_logging_from_env()
logger = get_logger(__name__)

def main():
    """Build the Pinecone vector index from JSONL files."""
    ap = argparse.ArgumentParser(description="Build Pinecone vector index from GroupMe messages")
    ap.add_argument("--input_glob", default="data/raw/*.jsonl", help="Glob pattern for input files")
    ap.add_argument("--window", type=int, default=1, help="Context window size")
    ap.add_argument("--clear", action="store_true", help="Clear existing index before adding new data")
    args = ap.parse_args()

    logger.info(f"Starting Pinecone index build with input_glob='{args.input_glob}', window={args.window}")

    files = sorted(glob.glob(args.input_glob))
    if not files:
        logger.error("No input files found. Put .jsonl dumps under data/raw/")
        return

    logger.info(f"Found {len(files)} input files: {files}")

    all_chunks = []
    total_records = 0

    for fp in files:
        logger.info(f"Processing file: {fp}")
        try:
            recs = load_jsonl(pathlib.Path(fp))
            total_records += len(recs)
            chunks = make_chunks_from_records(recs, window=args.window)
            all_chunks.extend(chunks)
            logger.info(f"{fp}: {len(recs)} records -> {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to process file {fp}: {str(e)}")
            continue

    logger.info(f"Total: {total_records} records -> {len(all_chunks)} chunks")

    if not all_chunks:
        logger.error("No chunks created. Check your input files.")
        return

    logger.info("Initializing RAG pipeline (Pinecone backend if configured)...")
    try:
        rag = RAGPipelineGM()

        # Clear existing index if requested
        if args.clear:
            logger.info("Clearing existing index...")
            rag.delete_all()

        # Show current index stats (only for Pinecone backend)
        if hasattr(rag, "get_stats"):
            stats = rag.get_stats()
            logger.info(f"Current index stats: {stats}")

        logger.info("Adding chunks to Pinecone vector database...")
        rag.add_messages(all_chunks)

        # Show final stats (only for Pinecone backend)
        if hasattr(rag, "get_stats"):
            final_stats = rag.get_stats()
            logger.info(f"Final index stats: {final_stats}")
        logger.info("Pinecone index build completed successfully!")

    except Exception as e:
        logger.error(f"Failed to build Pinecone index: {str(e)}")
        raise

if __name__ == "__main__":
    main()
