#!/usr/bin/env python3
"""
Test script to validate the Pinecone migration works correctly.
This script tests with a small sample of data to ensure everything is working.
"""

import os
import sys
import pathlib
from typing import List, Dict

# Add the server directory to the path so we can import our modules
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from server.ingestion.groupme_loader import load_jsonl, make_chunks_from_records
from server.rag.pipeline_groupme import RAGPipelineGM
from logging_config import setup_logging_from_env, get_logger

# Initialize logging
setup_logging_from_env()
logger = get_logger(__name__)

def test_basic_functionality():
    """Test basic Pinecone pipeline functionality."""
    logger.info("Testing basic Pinecone functionality...")

    try:
        # Initialize pipeline
        rag = RAGPipelineGM()
        logger.info("✅ Pipeline initialization successful")

        # Get initial stats
        stats = rag.get_stats()
        logger.info(f"✅ Index stats retrieved: {stats}")

        return rag
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {str(e)}")
        return None

def test_with_sample_data(rag: RAGPipelineGM, sample_size: int = 10):
    """Test with a small sample of real data."""
    logger.info(f"Testing with sample data (size: {sample_size})...")

    try:
        # Load a small sample of data
        data_path = pathlib.Path("data/raw/messages.jsonl")
        if not data_path.exists():
            logger.warning(f"❌ Sample data file not found: {data_path}")
            return False

        # Load only first few records
        records = []
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                import json
                records.append(json.loads(line.strip()))

        logger.info(f"✅ Loaded {len(records)} sample records")

        # Create chunks
        chunks = make_chunks_from_records(records, window=1)
        logger.info(f"✅ Created {len(chunks)} chunks from sample data")

        if not chunks:
            logger.warning("❌ No chunks created from sample data")
            return False

        # Clear any existing test data first
        logger.info("Clearing existing index data...")
        if hasattr(rag, "delete_all"):
            rag.delete_all()

        # Add chunks to Pinecone
        rag.add_messages(chunks)
        logger.info(f"✅ Successfully added {len(chunks)} chunks to Pinecone")

        # Test retrieval
        test_query = "anyone know about internships"
        results = rag.retrieve(test_query, k=3)
        logger.info(f"✅ Retrieved {len(results)} results for test query: '{test_query}'")

        # Show sample results
        for i, hit in enumerate(results[:2]):
            logger.info(f"Result {i+1} (score: {hit.score:.3f}): {hit.text[:100]}...")

        # Test generation
        response = rag.generate(test_query, k=3)
        logger.info(f"✅ Generated response ({len(response)} chars): {response[:100]}...")

        # Get final stats if available
        if hasattr(rag, "get_stats"):
            final_stats = rag.get_stats()
            logger.info(f"✅ Final index stats: {final_stats}")

        return True

    except Exception as e:
        logger.error(f"❌ Sample data test failed: {str(e)}")
        return False

def test_general_questions(rag: RAGPipelineGM):
    """Test general questions that don't need GroupMe context."""
    logger.info("Testing general questions...")

    try:
        test_queries = [
            "What are you?",
            "How do you work?",
            "What can you help me with?"
        ]

        for query in test_queries:
            response = rag.generate(query)
            logger.info(f"✅ Query: '{query}' -> Response: {response[:50]}...")

        return True

    except Exception as e:
        logger.error(f"❌ General questions test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Pinecone migration tests...")

    # Check environment variables
    required_env_vars = ["PINECONE_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        logger.error("Please set these in your .env file before running tests.")
        return False

    logger.info("✅ Required environment variables found")

    # Test basic functionality
    rag = test_basic_functionality()
    if not rag:
        return False

    # Test with sample data
    if not test_with_sample_data(rag, sample_size=10):
        return False

    # Test general questions
    if not test_general_questions(rag):
        return False

    logger.info("🎉 All tests passed! Pinecone migration is working correctly.")
    logger.info("You can now run the full migration with: python scripts/build_pinecone_index.py --clear")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
