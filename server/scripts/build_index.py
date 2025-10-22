import argparse, pathlib, glob
from ingestion.groupme_loader import load_jsonl, make_chunks_from_records
from ingestion.papernu_loader import make_context_from_papernu_data
from server.rag.pipeline import RAGPipeline
from logging_config import setup_logging_from_env, get_logger

# Initialize logging
setup_logging_from_env()
logger = get_logger(__name__)

def main():
    """Build vector indexes from GroupMe messages and PaperNU course data."""
    ap = argparse.ArgumentParser(description="Build vector index from GroupMe and PaperNU data")
    ap.add_argument("--input_glob", default="data/raw/*.jsonl", help="Glob pattern for GroupMe input files")
    ap.add_argument("--window", type=int, default=1, help="Context window size for GroupMe chunks")
    args = ap.parse_args()

    logger.info("=== Starting Index Build ===")
    
    # Initialize RAG pipeline
    logger.info("Initializing RAG pipeline...")
    rag = RAGPipeline()

    logger.info("--- Phase 1: Processing GroupMe messages ---")
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
        course_records = make_context_from_papernu_data()
        if course_records:
            logger.info(f"Loaded {len(course_records)} course records")
            logger.info("Adding course records to vector database...")
            rag.add_course_records(course_records)
            logger.info("PaperNU indexing complete")
        else:
            logger.warning("No course records loaded")
    except FileNotFoundError:
        logger.warning("PaperNU data file not found (data/raw/plan.json). Skipping course indexing.")
    except Exception as e:
        logger.error(f"Failed to process PaperNU data: {str(e)}")

    logger.info("=== Index Build Complete ===")


if __name__ == "__main__":
    main()
