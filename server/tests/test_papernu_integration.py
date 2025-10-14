import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.papernu_loader import make_context_from_papernu_data
from ingestion.models import FullCourseRecord
from rag.pipeline_papernu import RAGPipelinePaperNU
from pprint import pprint

def test_make_context_from_papernu_data():
    records: list[FullCourseRecord] = make_context_from_papernu_data()

    assert isinstance(records, list)
    assert len(records) > 100

def test_rag_pipeline_papernu():
    # Note: This test requires OpenAI API key and will make real API calls
    # Skip if OPENAI_API_KEY is not set
    if not os.getenv("OPENAI_API_KEY"):
        import pytest
        pytest.skip("OPENAI_API_KEY not set")
    
    pipeline = RAGPipelinePaperNU()
    query = "Hey, can you tell me about CS336 and its prereqs?"
    context = pipeline.build_context(query)
    print(f"Context for query {query}:")
    print(context)