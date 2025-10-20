from server.rag.pipeline import RAGPipelineGM
from logging_config import setup_logging_from_env, get_logger

# Initialize logging
setup_logging_from_env()
logger = get_logger(__name__)

def main():
    """Interactive chat interface for testing the RAG pipeline."""
    logger.info("Starting interactive chat session")

    try:
        rag = RAGPipelineGM()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
        return

    print("GroupMe RAG. Type 'exit' to quit.")
    logger.info("Chat session started")

    while True:
        try:
            q = input("\nYou: ").strip()
            if q.lower() in {"exit","quit"}:
                logger.info("User requested exit")
                break
            if not q:
                continue

            logger.info(f"User query: {q}")
            ans = rag.generate(q, k=6)
            print("\nBot:", ans)
            logger.info("Response generated successfully")

        except KeyboardInterrupt:
            logger.info("Chat session interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error during chat session: {str(e)}")
            print(f"\nError: {str(e)}")

    logger.info("Chat session ended")

if __name__ == "__main__":
    main()
