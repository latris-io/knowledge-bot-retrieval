#!/usr/bin/env python3
"""
Integration tests for problematic queries that were failing before enhanced retrieval.
Tests real-world scenarios that drove the need for enhanced retrieval capabilities.
"""

import pytest
import asyncio
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

class TestProblematicQueries:
    """Test suite for queries that were failing before enhanced retrieval"""
    
    # Known problematic queries from user reports
    PROBLEMATIC_QUERIES = [
        {
            "query": "does vishal have mulesoft experience",
            "expected_content": ["mulesoft", "vishal"],
            "query_type": "relationship",
            "description": "Relationship query about person's technology experience"
        },
        {
            "query": "when is the brentwood office open", 
            "expected_content": ["brentwood", "office", "hours", "open"],
            "query_type": "factual",
            "description": "Office hours query with location specificity"
        },
        {
            "query": "what technologies does marty know",
            "expected_content": ["marty", "technology", "skills"],
            "query_type": "relationship", 
            "description": "Skills/technology relationship query"
        },
        {
            "query": "vishal skills mulesoft",
            "expected_content": ["vishal", "mulesoft", "skills"],
            "query_type": "general",
            "description": "Keyword-style query without proper grammar"
        },
        {
            "query": "brentwood office hours",
            "expected_content": ["brentwood", "office", "hours"],
            "query_type": "factual",
            "description": "Keyword-style hours query"
        },
        {
            "query": "marty programming experience",
            "expected_content": ["marty", "programming", "experience"],
            "query_type": "relationship",
            "description": "Programming skills relationship query"
        }
    ]

    @pytest.mark.asyncio
    async def test_enhanced_retrieval_success_rate(self, retriever_service):
        """Test that enhanced retrieval succeeds where original failed"""
        
        success_count = 0
        total_queries = len(self.PROBLEMATIC_QUERIES)
        
        for test_case in self.PROBLEMATIC_QUERIES:
            query = test_case["query"]
            expected_content = test_case["expected_content"]
            
            try:
                # Test enhanced retrieval
                enhanced_retriever = await retriever_service.build_enhanced_retriever(
                    company_id=3,
                    bot_id=1, 
                    query=query,
                    k=10,
                    use_enhanced_search=True
                )
                
                results = enhanced_retriever.get_relevant_documents(query)
                
                # Check if we found relevant documents
                if results:
                    # Check if results contain expected content
                    found_content = set()
                    for doc in results:
                        content_lower = doc.page_content.lower()
                        for expected in expected_content:
                            if expected.lower() in content_lower:
                                found_content.add(expected.lower())
                    
                    # Success if we found at least half the expected content
                    if len(found_content) >= len(expected_content) / 2:
                        success_count += 1
                        print(f"✅ SUCCESS: '{query}' found {len(results)} docs with content: {found_content}")
                    else:
                        print(f"❌ PARTIAL: '{query}' found docs but missing expected content")
                else:
                    print(f"❌ FAILED: '{query}' found no documents")
                    
            except Exception as e:
                print(f"❌ ERROR: '{query}' failed with error: {e}")
                # Skip if ChromaDB unavailable but don't fail test
                if "chroma" in str(e).lower() or "connection" in str(e).lower():
                    pytest.skip(f"ChromaDB not available: {e}")
        
        # Success rate should be significantly improved
        success_rate = success_count / total_queries
        print(f"\nOverall success rate: {success_rate:.1%} ({success_count}/{total_queries})")
        
        # We expect at least 80% success rate with enhanced retrieval
        assert success_rate >= 0.8, f"Enhanced retrieval success rate {success_rate:.1%} below 80% threshold"

    @pytest.mark.asyncio
    async def test_query_type_classification(self, retriever_service):
        """Test that query types are correctly classified"""
        
        enhanced_retriever = retriever_service.enhanced_retriever
        
        for test_case in self.PROBLEMATIC_QUERIES:
            query = test_case["query"]
            expected_type = test_case["query_type"]
            
            actual_type = enhanced_retriever.classify_query_type(query)
            
            # Log classification for debugging
            print(f"Query: '{query}' | Expected: {expected_type} | Actual: {actual_type}")
            
            # Assertion with helpful error message
            assert actual_type == expected_type, \
                f"Query '{query}' classified as '{actual_type}', expected '{expected_type}'"

    @pytest.mark.asyncio
    async def test_adaptive_thresholds_for_problematic_queries(self, retriever_service):
        """Test that adaptive thresholds are appropriate for problematic queries"""
        
        enhanced_retriever = retriever_service.enhanced_retriever
        
        relationship_thresholds = []
        factual_thresholds = []
        
        for test_case in self.PROBLEMATIC_QUERIES:
            query = test_case["query"]
            query_type = test_case["query_type"]
            
            threshold = enhanced_retriever.get_adaptive_similarity_threshold(query)
            
            if query_type == "relationship":
                relationship_thresholds.append(threshold)
            elif query_type == "factual":
                factual_thresholds.append(threshold)
            
            # All thresholds should be reasonable
            assert 0.01 <= threshold <= 0.1, \
                f"Threshold {threshold} for '{query}' outside reasonable range [0.01, 0.1]"
        
        # Relationship queries should generally have lower thresholds
        if relationship_thresholds and factual_thresholds:
            avg_relationship = sum(relationship_thresholds) / len(relationship_thresholds)
            avg_factual = sum(factual_thresholds) / len(factual_thresholds)
            
            print(f"Average relationship threshold: {avg_relationship:.3f}")
            print(f"Average factual threshold: {avg_factual:.3f}")
            
            # Relationship should be lower for broader matching
            assert avg_relationship < avg_factual, \
                "Relationship queries should have lower thresholds than factual queries"

    @pytest.mark.asyncio
    async def test_entity_extraction_for_problematic_queries(self, retriever_service):
        """Test entity extraction works for our problematic queries"""
        
        enhanced_retriever = retriever_service.enhanced_retriever
        
        for test_case in self.PROBLEMATIC_QUERIES:
            query = test_case["query"]
            expected_content = test_case["expected_content"]
            
            entities = enhanced_retriever.extract_entities(query)
            concepts = enhanced_retriever.extract_concepts(query)
            
            # Should extract at least some entities/concepts
            assert len(entities) > 0 or len(concepts) > 0, \
                f"No entities or concepts extracted from '{query}'"
            
            # Should extract at least some expected content as entities or concepts
            all_extracted = entities + concepts
            found_expected = [item for item in expected_content 
                            if any(item.lower() in extracted.lower() for extracted in all_extracted)]
            
            assert len(found_expected) > 0, \
                f"None of expected content {expected_content} found in extracted {all_extracted} for '{query}'"

    @pytest.mark.asyncio
    async def test_semantic_expansion_for_problematic_queries(self, retriever_service):
        """Test semantic expansion generates useful alternatives"""
        
        enhanced_retriever = retriever_service.enhanced_retriever
        
        for test_case in self.PROBLEMATIC_QUERIES:
            query = test_case["query"]
            expected_content = test_case["expected_content"]
            
            # Test semantic expansion
            expanded_queries = await enhanced_retriever.expand_query_semantically(query)
            
            # Should generate alternatives
            assert len(expanded_queries) >= 2, \
                f"Only {len(expanded_queries)} queries generated for '{query}', expected at least 2"
            
            # Original query should be preserved
            assert query in expanded_queries, \
                f"Original query '{query}' not preserved in expansion"
            
            # Alternatives should contain key entities
            alternatives = [q for q in expanded_queries if q != query]
            for alt in alternatives:
                # At least one key entity should be preserved in each alternative
                preserved_entities = [entity for entity in expected_content 
                                    if entity.lower() in alt.lower()]
                assert len(preserved_entities) > 0, \
                    f"Alternative '{alt}' doesn't preserve any key entities from {expected_content}"

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"]) 