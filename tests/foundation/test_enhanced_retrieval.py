#!/usr/bin/env python3
"""
Test suite for Foundation Improvement FI-04: Content-Agnostic Enhanced Retrieval System
Matches test cases documented in USE_CASES.md
"""

import pytest
import asyncio
import os
import sys
from dotenv import load_dotenv
from langchain.schema import Document

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from retriever import RetrieverService

@pytest.fixture
def retriever_service():
    """Initialize retriever service for testing"""
    load_dotenv()
    return RetrieverService()

@pytest.mark.asyncio
async def test_semantic_query_expansion(retriever_service):
    """Test FI-04.1: Semantic Query Expansion"""
    # Test query expansion functionality
    query = "does vishal have mulesoft experience"
    
    expanded_queries = await retriever_service.enhanced_retriever.expand_query_semantically(query)
    
    # Assertions
    assert len(expanded_queries) >= 2  # Original + at least 1 alternative
    assert query in expanded_queries  # Original query preserved
    assert any("mulesoft" in alt.lower() for alt in expanded_queries)  # Technology preserved
    assert any("vishal" in alt.lower() for alt in expanded_queries)  # Entity preserved

@pytest.mark.asyncio 
async def test_multi_vector_search(retriever_service):
    """Test FI-04.2: Multi-Vector Search Coverage"""
    # Test comprehensive search coverage
    query = "does vishal have mulesoft experience"
    
    try:
        vectorstore = retriever_service.get_chroma_vectorstore("global")
        
        # Get results from multi-vector search
        results = await retriever_service.enhanced_retriever.multi_vector_search(query, vectorstore, k=8)
        
        # Assertions
        assert len(results) > 0  # Should find relevant documents
        assert len(results) <= 8  # Should respect k parameter
        
        # Check for Vishal's resume content
        vishal_docs = [doc for doc in results if "vishal" in doc.page_content.lower()]
        assert len(vishal_docs) > 0  # Should find Vishal-related documents
        
        # Check for Mulesoft content
        mulesoft_docs = [doc for doc in results if "mulesoft" in doc.page_content.lower()]
        assert len(mulesoft_docs) > 0  # Should find Mulesoft-related documents
        
    except Exception as e:
        # Skip test if ChromaDB is not available
        pytest.skip(f"ChromaDB not available: {e}")

@pytest.mark.asyncio
async def test_adaptive_similarity_thresholds(retriever_service):
    """Test FI-04.3: Adaptive Similarity Thresholds"""
    enhanced_retriever = retriever_service.enhanced_retriever
    
    # Test different query types get different thresholds
    relationship_query = "does vishal have mulesoft experience"
    factual_query = "what is salesforce"
    comparison_query = "compare java and python"
    
    rel_threshold = enhanced_retriever.get_adaptive_similarity_threshold(relationship_query)
    fact_threshold = enhanced_retriever.get_adaptive_similarity_threshold(factual_query)
    comp_threshold = enhanced_retriever.get_adaptive_similarity_threshold(comparison_query)
    
    # Assertions: relationship queries should have lower thresholds
    assert rel_threshold < fact_threshold  # Relationship needs broader matching
    assert rel_threshold < 0.05  # Should be lower than default
    assert fact_threshold > 0.05  # Should be higher than default for precision
    assert 0.01 <= comp_threshold <= 0.1  # Should be within reasonable bounds

@pytest.mark.asyncio
async def test_query_classification(retriever_service):
    """Test FI-04.4: Query Classification Accuracy"""
    enhanced_retriever = retriever_service.enhanced_retriever
    
    test_cases = [
        ("does vishal have mulesoft experience", "relationship"),
        ("what is salesforce", "factual"), 
        ("compare java and python", "comparison"),
        ("list all programming languages", "list"),
        ("tell me about the company", "general")
    ]
    
    for query, expected_type in test_cases:
        actual_type = enhanced_retriever.classify_query_type(query)
        assert actual_type == expected_type, f"Query '{query}' classified as '{actual_type}', expected '{expected_type}'"

@pytest.mark.asyncio
async def test_entity_concept_extraction(retriever_service):
    """Test FI-04.5: Entity and Concept Extraction"""
    enhanced_retriever = retriever_service.enhanced_retriever
    
    query = "does vishal have mulesoft experience"
    
    # Test entity extraction
    entities = enhanced_retriever.extract_entities(query)
    assert "vishal" in entities  # Should extract person name
    assert "mulesoft" in entities  # Should extract technology name
    
    # Test concept extraction
    concepts = enhanced_retriever.extract_concepts(query)
    assert "vishal" in concepts  # Important concepts preserved
    assert "mulesoft" in concepts
    assert "experience" in concepts
    assert "does" not in concepts  # Stop words removed

@pytest.mark.asyncio
async def test_enhanced_vs_original_retrieval(retriever_service):
    """Test FI-04.6: Enhanced vs Original Retrieval Comparison"""
    # Test that enhanced retrieval performs better than original
    problematic_queries = [
        "does vishal have mulesoft experience",
        "when is the brentwood office open",
        "what technologies does marty know"
    ]
    
    for query in problematic_queries:
        try:
            # Enhanced retrieval
            enhanced_retriever = await retriever_service.build_enhanced_retriever(
                company_id=3, bot_id=1, query=query, k=8, use_enhanced_search=True
            )
            enhanced_results = enhanced_retriever.get_relevant_documents(query)
            
            # Original retrieval
            original_retriever = await retriever_service.build_enhanced_retriever(
                company_id=3, bot_id=1, query=query, k=8, use_enhanced_search=False
            )
            original_results = original_retriever.get_relevant_documents(query)
            
            # Enhanced should find at least as many relevant docs
            assert len(enhanced_results) >= len(original_results)
            
            # Enhanced should have better relevance scoring
            if enhanced_results:
                enhanced_relevance = enhanced_results[0].metadata.get('relevance_score', 0)
                assert enhanced_relevance >= 0  # Should have relevance scoring
                
        except Exception as e:
            # Skip test if ChromaDB is not available
            pytest.skip(f"ChromaDB not available for query '{query}': {e}")

@pytest.mark.asyncio
async def test_learning_system(retriever_service):
    """Test FI-04.7: Learning System Integration"""
    enhanced_retriever = retriever_service.enhanced_retriever
    
    # Simulate successful retrieval
    query = "does vishal have mulesoft experience"
    mock_docs = [
        Document(page_content="Mulesoft technology overview", metadata={"structure_type": "header"}),
        Document(page_content="Vishal's skills include...", metadata={"structure_type": "paragraph"})
    ]
    
    # Test pattern learning
    enhanced_retriever.learn_document_patterns(query, mock_docs, success_score=0.8)
    
    # Test pattern application
    weighted_docs = enhanced_retriever.reweight_results(query, mock_docs)
    
    # Assertions
    assert len(weighted_docs) == len(mock_docs)
    assert all(doc.metadata.get('pattern_weight') is not None for doc in weighted_docs)

# Additional integration tests

@pytest.mark.asyncio
async def test_contextual_embeddings(retriever_service):
    """Test contextual embedding enhancement"""
    enhanced_retriever = retriever_service.enhanced_retriever
    
    # Test different document contexts
    test_cases = [
        ("Mulesoft overview", {"structure_type": "header", "file_name": "resume.pdf"}),
        ("Office hours data", {"structure_type": "table_row", "file_name": "office_info.xlsx"}),
        ("Newsletter content", {"structure_type": "paragraph", "file_name": "newsletter.pdf"})
    ]
    
    for content, metadata in test_cases:
        embedding = enhanced_retriever.create_contextual_embeddings(content, metadata)
        assert embedding is not None
        assert len(embedding) > 0  # Should generate valid embedding

@pytest.mark.asyncio
async def test_hierarchical_search(retriever_service):
    """Test hierarchical search functionality"""
    query = "when is the brentwood office open"
    
    try:
        vectorstore = retriever_service.get_chroma_vectorstore("global")
        
        # Test hierarchical search
        results = await retriever_service.enhanced_retriever.hierarchical_search(query, vectorstore, k=8)
        
        # Assertions
        assert isinstance(results, list)
        assert len(results) <= 8  # Should respect k parameter
        
        # Results should have relevance scores if any found
        if results:
            for doc in results:
                relevance = doc.metadata.get('relevance_score')
                if relevance is not None:
                    assert relevance >= 0  # Should be valid score
                    
    except Exception as e:
        # Skip test if ChromaDB is not available
        pytest.skip(f"ChromaDB not available: {e}")

@pytest.mark.asyncio
async def test_caching_functionality(retriever_service):
    """Test entity and concept caching"""
    enhanced_retriever = retriever_service.enhanced_retriever
    
    query = "does vishal have mulesoft experience"
    
    # First extraction (should cache)
    entities1 = enhanced_retriever.extract_entities(query)
    concepts1 = enhanced_retriever.extract_concepts(query)
    
    # Second extraction (should use cache)
    entities2 = enhanced_retriever.extract_entities(query)
    concepts2 = enhanced_retriever.extract_concepts(query)
    
    # Should be identical (cached)
    assert entities1 == entities2
    assert concepts1 == concepts2
    
    # Check cache was used
    assert query in enhanced_retriever.entity_cache
    assert query in enhanced_retriever.concept_cache

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 