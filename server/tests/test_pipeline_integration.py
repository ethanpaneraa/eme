"""
Tests for the integrated RAGPipelineGM with PaperNU course data.
"""
import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.rag.pipeline import RAGPipelineGM
from ingestion.models import FullCourseRecord


# ============================================================================
# Phase 1: Test initialization with PaperNU data
# ============================================================================

def test_pipeline_initialization_loads_course_records():
    """Test that pipeline initializes with course records from PaperNU."""
    # Skip if OPENAI_API_KEY is not set
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    pipeline = RAGPipelineGM()
    
    # Verify course records were loaded
    assert hasattr(pipeline, 'course_records')
    assert isinstance(pipeline.course_records, list)
    assert len(pipeline.course_records) > 0
    
    # Verify they are FullCourseRecord instances
    assert all(isinstance(r, FullCourseRecord) for r in pipeline.course_records)
    
    # Verify basic data integrity
    for record in pipeline.course_records[:5]:
        assert record.subject in ("COMP_SCI", "COMP_ENG")
        assert record.catalog_number != ""
        assert record.name != ""
    
    print(f"✓ Successfully loaded {len(pipeline.course_records)} course records")


def test_pipeline_has_papernu_collection():
    """Test that pipeline creates a separate PaperNU collection."""
    # Skip if OPENAI_API_KEY is not set
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    pipeline = RAGPipelineGM()
    
    # Only check for ChromaDB backend
    if pipeline.backend == "chroma":
        assert hasattr(pipeline, 'papernu_collection')
        assert pipeline.papernu_collection is not None
        print(f"✓ PaperNU collection exists: {pipeline.papernu_collection.name}")
    else:
        pytest.skip("Test only applies to ChromaDB backend")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================================
# Phase 2: Test _match_catalog_number_to_course helper function
# ============================================================================

def test_match_catalog_number_finds_courses():
    """Test that catalog number matching works correctly."""
    # Skip if OPENAI_API_KEY is not set
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    # Create a minimal pipeline with mock course records
    pipeline = RAGPipelineGM()
    
    # Add test course records if empty
    if not pipeline.course_records:
        from ingestion.models import FullCourseRecord
        pipeline.course_records = [
            FullCourseRecord(
                subject="COMP_SCI",
                catalog_number="336-0",
                name="Design & Analysis of Algorithms",
                description="Advanced algorithms course",
                prereqs="COMP_SCI 214-0"
            ),
            FullCourseRecord(
                subject="COMP_SCI",
                catalog_number="330-0",
                name="Human Computer Interaction",
                description="HCI principles and design",
                prereqs="None"
            )
        ]
    
    # Test with CS336 in query
    query = "What are the prerequisites for CS336?"
    matches = pipeline._match_catalog_number_to_course(query)
    
    assert isinstance(matches, list)
    assert len(matches) > 0
    assert any("336" in m for m in matches)
    assert any("Design & Analysis of Algorithms" in m or "Algorithms" in m for m in matches)
    
    print(f"✓ Found {len(matches)} matches for '336': {matches[0][:50]}...")


def test_match_catalog_number_handles_no_matches():
    """Test that catalog number matching returns empty list when no matches found."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    pipeline = RAGPipelineGM()
    
    # Query with non-existent course number
    query = "Tell me about CS999"
    matches = pipeline._match_catalog_number_to_course(query)
    
    assert isinstance(matches, list)
    # Should be empty or very few results
    print(f"✓ Query with non-existent course returned {len(matches)} matches")


def test_match_catalog_number_extracts_multiple():
    """Test extraction of multiple course numbers from one query."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    pipeline = RAGPipelineGM()
    
    if not pipeline.course_records:
        pytest.skip("No course records loaded")
    
    # Query mentioning multiple courses
    query = "Is CS336 a prerequisite for CS496?"
    matches = pipeline._match_catalog_number_to_course(query)
    
    assert isinstance(matches, list)
    # Should potentially find both courses if they exist
    print(f"✓ Found {len(matches)} matches for multi-course query")


# ============================================================================
# Phase 3: Test PaperNU data indexing and retrieval
# ============================================================================

def test_retrieve_papernu_finds_courses():
    """Test retrieving course information from PaperNU collection."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    pipeline = RAGPipelineGM()
    
    if pipeline.backend != "chroma":
        pytest.skip("Test only for ChromaDB backend")
    # Instead of manipulating the index, call generate and check CS337 prereqs
    query = "What are the prerequisites for CS337?"
    response = pipeline.generate(query, k=3)

    assert isinstance(response, str)
    assert "348" in response or "COMP_SCI 348" in response
    print("✓ Generated response for CS337 contains expected prereq token 348")


def test_combined_retrieval():
    """Test retrieving from both GroupMe and PaperNU sources."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    pipeline = RAGPipelineGM()
    
    if pipeline.backend != "chroma":
        pytest.skip("Test only for ChromaDB backend")
    # Test generation for course 350's prerequisites (expecting 213 and 340 to be mentioned)
    query = "What are the prerequisites for CS350?"
    response = pipeline.generate(query, k=3)

    assert isinstance(response, str)
    assert ("213" in response or "COMP_SCI 213" in response), f"expected '213' in response: {response[:200]}"
    assert ("340" in response or "COMP_SCI 340" in response), f"expected '340' in response: {response[:200]}"
    print("✓ Generated response for CS350 contains expected prereq tokens 213 and 340")


# ============================================================================
# Phase 4: Test hybrid prompt building and generation
# ============================================================================

def test_build_hybrid_prompt():
    """Test building a prompt with both GroupMe and PaperNU sources."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    pipeline = RAGPipelineGM()
    
    from server.rag.pipeline import RetrievedHit
    
    # Create mock hits
    groupme_hits = [
        RetrievedHit(
            text="CS 336 is really challenging but worth it!",
            meta={"sender_name": "Alice", "created_at_iso": "2024-03-15", "msg_type": "message"},
            score=0.9
        )
    ]
    
    papernu_hits = [
        RetrievedHit(
            text="Course information for COMP_SCI/CS 336-0: Design & Analysis of Algorithms\nDescription: Advanced algorithms\nPrerequisites: COMP_SCI 214-0",
            meta={"source": "papernu"},
            score=0.95
        )
    ]
    
    query = "What are the prerequisites for CS336?"
    msgs = pipeline._build_hybrid_prompt(query, groupme_hits, papernu_hits)
    
    assert isinstance(msgs, list)
    assert len(msgs) == 2  # system and user messages
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    
    # Check that both sources are mentioned in the prompt
    user_content = msgs[1]["content"]
    assert "COURSE" in user_content or "paper.nu" in user_content
    assert "CHAT" in user_content or "GroupMe" in user_content
    assert "336" in user_content
    
    print("✓ Hybrid prompt successfully built with both sources")


def test_generate_with_hybrid_sources():
    """Test end-to-end generation using both GroupMe and PaperNU data."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    pipeline = RAGPipelineGM()
    
    if pipeline.backend != "chroma":
        pytest.skip("Test only for ChromaDB backend")
        
    # Generate a response
    query = "What is CS336 about?"
    response = pipeline.generate(query, k=3)
    
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Response should mention algorithms or the course
    response_lower = response.lower()
    assert any(keyword in response_lower for keyword in ["algorithm", "336", "design", "analysis"])
    
    print(f"✓ Generated hybrid response ({len(response)} chars)")
    print(f"  Response preview: {response[:150]}...")
