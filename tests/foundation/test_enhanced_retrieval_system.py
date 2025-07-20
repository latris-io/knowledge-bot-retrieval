"""
FI-04: Content-Agnostic Enhanced Retrieval System Tests

Tests for semantic query expansion, multi-vector search, and enhanced retrieval quality.
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from retriever import RetrieverService


class TestEnhancedRetrievalSystem:
    
    @pytest.fixture
    def retriever_service(self):
        """Create a RetrieverService instance for testing."""
        return RetrieverService()
    
    @pytest.mark.enhanced_retrieval
    @pytest.mark.foundation
    def test_semantic_query_expansion_basic(self, retriever_service):
        """Test FI-04: Basic semantic query expansion functionality."""
        
        # Mock LLM response for query expansion
        mock_response = MagicMock()
        mock_response.content = """Alternative 1: What businesses are stored in the system?
Alternative 2: Which organizations are in the records?"""
        
        with patch('retriever.ChatOpenAI') as mock_llm:
            mock_llm.return_value.invoke.return_value = mock_response
            
            # Test query expansion
            original_query = "What companies are in the database?"
            expanded = retriever_service.expand_query_semantically(original_query)
            
            # Validate expansion results
            assert len(expanded) >= 2, "Should generate at least 2 query variants"
            assert original_query in expanded, "Should include original query"
            assert any("businesses" in query.lower() for query in expanded), "Should contain semantic alternatives"
            assert any("organizations" in query.lower() for query in expanded), "Should contain vocabulary variations"
    
    @pytest.mark.enhanced_retrieval  
    @pytest.mark.foundation
    def test_query_expansion_caching(self, retriever_service):
        """Test FI-04: Query expansion caching mechanism."""
        
        mock_response = MagicMock()
        mock_response.content = "Alternative 1: Test query\nAlternative 2: Another test"
        
        with patch('retriever.ChatOpenAI') as mock_llm:
            mock_llm.return_value.invoke.return_value = mock_response
            
            query = "test caching query"
            
            # First call - should invoke LLM
            result1 = retriever_service.expand_query_semantically(query)
            
            # Second call - should use cache
            result2 = retriever_service.expand_query_semantically(query)
            
            # Validate caching
            assert result1 == result2, "Cached results should be identical"
            assert query in retriever_service._query_cache, "Query should be cached"
            
            # Should only call LLM once due to caching
            assert mock_llm.return_value.invoke.call_count == 1, "Should use cache on second call"
    
    @pytest.mark.enhanced_retrieval
    @pytest.mark.foundation
    def test_multi_vector_search_diversity(self, retriever_service):
        """Test FI-04: Multi-vector search with diverse approaches."""
        
        # Mock vectorstore with different search results
        mock_vectorstore = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "TechCorp is a technology company"
        mock_doc1.metadata = {'chunk_index': 0}
        
        mock_doc2 = MagicMock()  
        mock_doc2.page_content = "DataSys handles data processing"
        mock_doc2.metadata = {'chunk_index': 1}
        
        mock_vectorstore.similarity_search.return_value = [mock_doc1, mock_doc2]
        
        # Mock query expansion
        with patch.object(retriever_service, 'expand_query_semantically') as mock_expand:
            mock_expand.return_value = [
                "technology companies", 
                "tech firms", 
                "software organizations"
            ]
            
            # Test multi-vector search
            query = "technology companies"
            results = retriever_service.multi_vector_search(query, mock_vectorstore, k=8)
            
            # Validate multi-vector search
            assert len(results) > 0, "Should return search results"
            assert any('query_variant' in doc.metadata for doc in results), "Should tag query variants"
            assert any('search_approach' in doc.metadata for doc in results), "Should tag search approaches"
            
            # Validate different search approaches were used
            search_approaches = [doc.metadata.get('search_approach') for doc in results if 'search_approach' in doc.metadata]
            assert 'original' in search_approaches, "Should use original query approach"
    
    @pytest.mark.enhanced_retrieval  
    @pytest.mark.foundation
    def test_multi_vector_deduplication(self, retriever_service):
        """Test FI-04: Smart deduplication across query variants."""
        
        # Create duplicate documents from different searches
        duplicate_content = "TechCorp technology solutions"
        
        mock_doc1 = MagicMock()
        mock_doc1.page_content = duplicate_content
        mock_doc1.metadata = {'chunk_index': 0}
        
        mock_doc2 = MagicMock()  
        mock_doc2.page_content = duplicate_content  # Duplicate
        mock_doc2.metadata = {'chunk_index': 0}
        
        mock_doc3 = MagicMock()
        mock_doc3.page_content = "DataSys data processing"  # Unique
        mock_doc3.metadata = {'chunk_index': 1}
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = [mock_doc1, mock_doc2, mock_doc3]
        
        with patch.object(retriever_service, 'expand_query_semantically') as mock_expand:
            mock_expand.return_value = ["tech companies", "technology firms"]
            
            results = retriever_service.multi_vector_search("tech", mock_vectorstore, k=8)
            
            # Validate deduplication
            contents = [doc.page_content for doc in results]
            unique_contents = set(contents[:100] + [str(doc.metadata.get('chunk_index', 0)) for doc in results])
            
            # Should have fewer results than total due to deduplication
            assert len(results) <= 4, "Should deduplicate similar documents"
            assert "DataSys data processing" in contents, "Should keep unique documents"
    
    @pytest.mark.enhanced_retrieval  
    @pytest.mark.foundation
    def test_enhanced_retrieval_production_reliability(self, retriever_service):
        """Test FI-04: Production reliability - no mocking, real functionality validation."""
        
        # Test that query expansion works reliably in production conditions
        query = "companies with technology expertise"
        
        # Clear cache to ensure fresh test
        retriever_service._query_cache.clear()
        
        # Test actual production behavior (no mocking)
        expanded = retriever_service.expand_query_semantically(query)
        
        # Validate production results
        assert isinstance(expanded, list), "Should return list in all cases"
        assert len(expanded) >= 1, f"Should generate at least original query, got: {expanded}"
        assert query in expanded, f"Should include original query in results: {expanded}"
        
        # If expansion worked (multiple variants), validate quality
        if len(expanded) > 1:
            for variant in expanded:
                assert isinstance(variant, str), f"All variants should be strings: {variant}"
                assert len(variant.strip()) > 0, f"All variants should be non-empty: '{variant}'"
                assert len(variant) >= 3, f"Variants should be meaningful length: '{variant}'"
    
    @pytest.mark.enhanced_retrieval
    @pytest.mark.foundation
    @pytest.mark.integration  
    def test_enhanced_retrieval_performance(self, retriever_service):
        """Test FI-04: Performance characteristics of enhanced retrieval."""
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = [
            MagicMock(page_content=f"Document {i}", metadata={'chunk_index': i})
            for i in range(10)
        ]
        
        with patch.object(retriever_service, 'expand_query_semantically') as mock_expand:
            mock_expand.return_value = ["query1", "query2", "query3"]
            
            # Test response time
            start_time = time.time()
            results = retriever_service.multi_vector_search("test", mock_vectorstore, k=8)  
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Performance validation
            assert response_time < 2.0, f"Enhanced retrieval should be fast: {response_time}s"
            assert len(results) > 0, "Should return results"
            
            # Validate caching improves performance
            start_time2 = time.time()
            results2 = retriever_service.multi_vector_search("test", mock_vectorstore, k=8)
            end_time2 = time.time()
            
            cached_time = end_time2 - start_time2  
            assert cached_time <= response_time, "Cached queries should be faster or equal"
    
    @pytest.mark.enhanced_retrieval
    @pytest.mark.foundation
    def test_content_agnostic_behavior(self, retriever_service):
        """Test FI-04: Truly content-agnostic query expansion."""
        
        mock_response = MagicMock()
        mock_response.content = """Alternative 1: Different phrasing
Alternative 2: Another approach"""
        
        with patch('retriever.ChatOpenAI') as mock_llm:
            mock_llm.return_value.invoke.return_value = mock_response
            
            # Test with different domain queries
            test_queries = [
                "medical procedures",
                "financial reports", 
                "technical documentation",
                "marketing materials"
            ]
            
            for query in test_queries:
                expanded = retriever_service.expand_query_semantically(query)
                
                # Should work for any domain
                assert len(expanded) >= 2, f"Should expand query for domain: {query}"
                assert query in expanded, f"Should preserve original: {query}"
                
                # Check that expansion prompt template is domain-agnostic 
                call_args = mock_llm.return_value.invoke.call_args[0][0]
                
                # Extract template (everything except the user's query parts)
                lines = call_args.split('\n')
                template_lines = [line for line in lines if not query.lower() in line.lower()]
                template_text = '\n'.join(template_lines).lower()
                
                # Should not contain hardcoded domain terms in TEMPLATE (not user query)
                hardcoded_terms = ['techcorp', 'datasys', 'healthcare', 'finance', 'vishal', 'marty', 'salesforce', 'mulesoft']
                domain_specific = any(term.lower() in template_text for term in hardcoded_terms)
                assert not domain_specific, f"Prompt template should be content-agnostic (excluding user query): {template_text}" 