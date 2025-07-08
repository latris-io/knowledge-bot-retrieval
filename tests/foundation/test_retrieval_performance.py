#!/usr/bin/env python3
"""
Performance tests for enhanced retrieval system.
Tests latency, throughput, and resource usage.
"""

import pytest
import asyncio
import time
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from retriever import RetrieverService

@pytest.fixture
def retriever_service():
    """Initialize retriever service for testing"""
    load_dotenv()
    return RetrieverService()

@pytest.mark.asyncio
async def test_enhanced_retrieval_latency(retriever_service):
    """Test that enhanced retrieval meets latency requirements (<200ms overhead)"""
    
    query = "does vishal have mulesoft experience"
    
    try:
        # Test original retrieval time
        start_time = time.time()
        original_retriever = await retriever_service.build_enhanced_retriever(
            company_id=3, bot_id=1, query=query, k=8, use_enhanced_search=False
        )
        original_results = original_retriever.get_relevant_documents(query)
        original_time = time.time() - start_time
        
        # Test enhanced retrieval time  
        start_time = time.time()
        enhanced_retriever = await retriever_service.build_enhanced_retriever(
            company_id=3, bot_id=1, query=query, k=8, use_enhanced_search=True
        )
        enhanced_results = enhanced_retriever.get_relevant_documents(query)
        enhanced_time = time.time() - start_time
        
        # Calculate overhead
        overhead = enhanced_time - original_time
        
        print(f"Original retrieval: {original_time:.3f}s")
        print(f"Enhanced retrieval: {enhanced_time:.3f}s")
        print(f"Overhead: {overhead:.3f}s")
        
        # Enhanced should have acceptable overhead (<200ms)
        assert overhead < 0.2, f"Enhanced retrieval overhead {overhead:.3f}s exceeds 200ms limit"
        
        # Enhanced should still find results
        assert len(enhanced_results) >= len(original_results), \
            "Enhanced retrieval found fewer results than original"
            
    except Exception as e:
        if "chroma" in str(e).lower():
            pytest.skip(f"ChromaDB not available: {e}")
        else:
            raise

@pytest.mark.asyncio
async def test_semantic_expansion_performance(retriever_service):
    """Test semantic expansion performance"""
    
    queries = [
        "does vishal have mulesoft experience",
        "when is the brentwood office open", 
        "what technologies does marty know"
    ]
    
    for query in queries:
        start_time = time.time()
        expanded = await retriever_service.enhanced_retriever.expand_query_semantically(query)
        expansion_time = time.time() - start_time
        
        print(f"Expansion time for '{query}': {expansion_time:.3f}s")
        
        # Should complete within reasonable time
        assert expansion_time < 5.0, f"Semantic expansion took {expansion_time:.3f}s, exceeds 5s limit"
        assert len(expanded) >= 2, "Should generate at least one alternative"

@pytest.mark.asyncio
async def test_caching_performance(retriever_service):
    """Test that caching improves performance"""
    
    enhanced_retriever = retriever_service.enhanced_retriever
    query = "does vishal have mulesoft experience"
    
    # First extraction (no cache)
    start_time = time.time()
    entities1 = enhanced_retriever.extract_entities(query)
    concepts1 = enhanced_retriever.extract_concepts(query)
    first_time = time.time() - start_time
    
    # Second extraction (with cache)
    start_time = time.time()
    entities2 = enhanced_retriever.extract_entities(query)
    concepts2 = enhanced_retriever.extract_concepts(query)
    second_time = time.time() - start_time
    
    print(f"First extraction: {first_time:.4f}s")
    print(f"Cached extraction: {second_time:.4f}s")
    
    # Cached should be faster
    assert second_time <= first_time, "Cached extraction should be faster or equal"
    
    # Results should be identical
    assert entities1 == entities2
    assert concepts1 == concepts2

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 