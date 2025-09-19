import argparse, pathlib, glob
from ingestion.loader import load_jsonl, make_chunks_from_records
from rag.pipeline import RAGPipelineGM
from logging_config import setup_logging_from_env, get_logger

# Initialize logging
setup_logging_from_env()
logger = get_logger(__name__)

def main():
    """Build the vector index from JSONL files."""
    ap = argparse.ArgumentParser(description="Build vector index from GroupMe messages")
    ap.add_argument("--input_glob", default="data/raw/*.jsonl", help="Glob pattern for input files")
    ap.add_argument("--window", type=int, default=1, help="Context window size")
    args = ap.parse_args()

    logger.info(f"Starting index build with input_glob='{args.input_glob}', window={args.window}")

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

    logger.info("Initializing RAG pipeline...")
    try:
        rag = RAGPipelineGM()
        logger.info("Adding chunks to vector database...")
        rag.add_messages(all_chunks)
        logger.info("Index build completed successfully!")
    except Exception as e:
        logger.error(f"Failed to build index: {str(e)}")
        raise

if __name__ == "__main__":
    main()
