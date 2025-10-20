import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.papernu_loader import make_context_from_papernu_data
from ingestion.models import FullCourseRecord
from rag.pipeline_groupme import RAGPipelineGM
from pprint import pprint

def test_make_context_from_papernu_data():
    records: list[FullCourseRecord] = make_context_from_papernu_data()

    print("Preview of loaded PaperNU course records:")
    for record in records[:2]:
        pprint(record)

    assert isinstance(records, list)
    assert len(records) > 100

def test_rag_pipeline_papernu():
    # Note: This test requires OpenAI API key and will make real API calls
    # Skip if OPENAI_API_KEY is not set
    if not os.getenv("OPENAI_API_KEY"):
        import pytest
        pytest.skip("OPENAI_API_KEY not set")
    
    pipeline = RAGPipelineGM()
    query = "What are the prereqs for CS337?"
    response = pipeline.generate(query)

    assert all(text in response for text in ["348"]), f"Expected prereq course numbers 348 not found in context: {response}"

    print(f"Generated context for query '{query}':\n{response}")
