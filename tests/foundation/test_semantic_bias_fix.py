"""
FI-05: Content-Agnostic Semantic Bias Fix Tests

Tests for universal term importance analysis and bias-free document re-ranking.
"""

import pytest
from unittest.mock import MagicMock
from retriever import RetrieverService


class TestSemanticBiasFix:
    
    @pytest.fixture
    def retriever_service(self):
        """Create a RetrieverService instance for testing."""
        return RetrieverService()
    
    @pytest.mark.semantic_bias
    @pytest.mark.foundation
    def test_query_term_importance_analysis(self, retriever_service):
        """Test FI-05: Universal term importance scoring."""
        
        # Test various query patterns
        test_cases = [
            {
                'query': 'TechCorp software experience',
                'expected_important': ['TechCorp', 'software', 'experience'],
                'expected_scores': {'TechCorp': '>1.0', 'software': '>1.0'}  # Capitalized and longer terms
            },
            {
                'query': 'healthcare technology solutions',
                'expected_important': ['healthcare', 'technology', 'solutions'],
                'expected_scores': {'healthcare': '>1.2', 'technology': '>1.2'}  # Longer terms
            },
            {
                'query': 'What company has API integration capabilities?',
                'expected_important': ['company', 'API', 'integration', 'capabilities'],
                'expected_scores': {'API': '>1.0', 'capabilities': '>1.0'}  # Capitalized, longer
            }
        ]
        
        for case in test_cases:
            importance_scores = retriever_service.analyze_query_term_importance(case['query'])
            
            # Validate important terms are identified
            for term in case['expected_important']:
                assert term.lower() in [t.lower() for t in importance_scores.keys()], \
                    f"Important term '{term}' should be identified in: {case['query']}"
            
            # Validate scoring logic  
            for term, expected in case['expected_scores'].items():
                score = importance_scores.get(term.lower(), 0)
                if expected == '>1.0':
                    assert score > 1.0, f"Term '{term}' should have score >1.0, got {score}"
                elif expected == '>1.2':
                    assert score > 1.2, f"Term '{term}' should have score >1.2, got {score}"
    
    @pytest.mark.semantic_bias
    @pytest.mark.foundation  
    def test_universal_importance_heuristics(self, retriever_service):
        """Test FI-05: Universal (content-agnostic) importance heuristics."""
        
        # Test length-based scoring
        query = "AI vs ArtificialIntelligence"
        scores = retriever_service.analyze_query_term_importance(query)
        
        ai_score = scores.get('ai', 0) 
        long_score = scores.get('artificialintelligence', 0)
        assert long_score > ai_score, "Longer terms should have higher importance"
        
        # Test capitalization detection  
        query = "compare TechCorp with techcorp solutions"
        scores = retriever_service.analyze_query_term_importance(query)
        
        # Both should be scored but capitalized form should get boost
        assert 'techcorp' in scores, "Should identify lowercase term"
        assert scores['techcorp'] > 1.0, "Proper nouns should get capitalization boost"
        
        # Test position-based importance
        query = "TechCorp provides excellent customer service solutions"
        scores = retriever_service.analyze_query_term_importance(query)
        
        first_term_score = scores.get('techcorp', 0)
        last_term_score = scores.get('solutions', 0)
        middle_term_score = scores.get('customer', 0)
        
        # First and last terms should get position boost
        assert first_term_score > middle_term_score, "First term should get position boost"
        assert last_term_score >= middle_term_score, "Last term should get position boost"
    
    @pytest.mark.semantic_bias
    @pytest.mark.foundation
    def test_content_agnostic_analysis(self, retriever_service):
        """Test FI-05: No domain-specific hardcoded patterns."""
        
        # Test across different domains to ensure no hardcoded bias
        domain_queries = [
            "medical imaging technology",
            "financial trading algorithms", 
            "automotive manufacturing process",
            "educational curriculum design",
            "agricultural sustainability practices"
        ]
        
        for query in domain_queries:
            scores = retriever_service.analyze_query_term_importance(query)
            
            # Validate universal principles apply
            assert len(scores) > 0, f"Should analyze terms for: {query}"
            
            # Check no hardcoded domain-specific boosting
            for term, score in scores.items():
                # Score should be based on universal heuristics only
                assert score >= 1.0, f"All terms should have base score ≥1.0: {term}={score}"
                assert score <= 3.0, f"No excessive hardcoded boosting: {term}={score}"
                
            # Longer terms should consistently score higher across domains
            sorted_terms = sorted(scores.items(), key=lambda x: len(x[0]), reverse=True)
            if len(sorted_terms) >= 2:
                longest_term = sorted_terms[0]
                shorter_term = sorted_terms[-1]
                
                if len(longest_term[0]) > len(shorter_term[0]) + 2:  # Significant length difference
                    assert longest_term[1] >= shorter_term[1], \
                        f"Longer term should score higher: {longest_term} vs {shorter_term}"
    
    @pytest.mark.semantic_bias
    @pytest.mark.foundation  
    def test_document_reranking_by_importance(self, retriever_service):
        """Test FI-05: Document re-ranking based on important terms."""
        
        query = "TechCorp software integration"
        
        # Create test documents with different relevance
        doc1 = MagicMock()
        doc1.page_content = "TechCorp specializes in software integration solutions for enterprises"
        doc1.metadata = {}
        
        doc2 = MagicMock() 
        doc2.page_content = "DataSys provides data processing services for various clients"
        doc2.metadata = {}
        
        doc3 = MagicMock()
        doc3.page_content = "Software integration is important for TechCorp business operations"  
        doc3.metadata = {}
        
        documents = [doc2, doc1, doc3]  # Intentionally out of relevance order
        
        # Test re-ranking
        reranked = retriever_service.rerank_by_term_importance(query, documents)
        
        # Validate re-ranking
        assert len(reranked) == 3, "Should return all documents"
        
        # Check importance scores were added
        for doc in reranked:
            assert 'importance_score' in doc.metadata, "Should add importance scores"
            assert 'term_matches' in doc.metadata, "Should count term matches"
        
        # Validate ranking quality
        scores = [doc.metadata['importance_score'] for doc in reranked]
        
        # Documents with more important terms should rank higher
        doc1_idx = reranked.index(doc1)  # Contains TechCorp + software + integration
        doc2_idx = reranked.index(doc2)  # Contains none of the important terms
        
        assert doc1_idx < doc2_idx, "More relevant document should rank higher"
        
        # Verify term matching logic
        doc1_matches = reranked[doc1_idx].metadata['term_matches']
        doc2_matches = reranked[doc2_idx].metadata['term_matches'] 
        
        assert doc1_matches > doc2_matches, "Better matching document should have more term matches"
    
    @pytest.mark.semantic_bias  
    @pytest.mark.foundation
    def test_length_normalization_prevents_bias(self, retriever_service):
        """Test FI-05: Length normalization prevents document length bias."""
        
        query = "technology solutions"
        
        # Create documents of different lengths with same relevance
        short_doc = MagicMock()
        short_doc.page_content = "technology solutions"  # Short but perfectly relevant
        short_doc.metadata = {}
        
        long_doc = MagicMock()
        long_doc.page_content = (
            "Our comprehensive business methodology incorporates various technology solutions "
            "alongside traditional approaches to ensure optimal client satisfaction through "
            "detailed analysis and extensive consultation processes that deliver results "
            "through systematic implementation of technology solutions for enterprise clients"
        )  # Long with repetition
        long_doc.metadata = {}
        
        documents = [long_doc, short_doc]
        
        reranked = retriever_service.rerank_by_term_importance(query, documents)
        
        # Validate length normalization
        short_score = reranked[1].metadata['importance_score'] if reranked[1] == short_doc else reranked[0].metadata['importance_score']
        long_score = reranked[0].metadata['importance_score'] if reranked[0] == long_doc else reranked[1].metadata['importance_score']
        
        # Short, highly relevant document should not be penalized
        assert short_score > 0, "Short relevant document should get positive score"
        
        # Length normalization should prevent long document from dominating just due to repetition
        ratio = long_score / short_score if short_score > 0 else float('inf')
        assert ratio < 5.0, f"Long document should not dominate due to length alone: ratio={ratio}"
    
    @pytest.mark.semantic_bias
    @pytest.mark.foundation
    def test_bias_correction_cross_domain(self, retriever_service):
        """Test FI-05: Bias correction works across different domains."""
        
        # Test person/technology attribution across domains  
        test_cases = [
            {
                'query': 'John healthcare experience',
                'docs': [
                    "John has extensive healthcare experience in clinical settings",
                    "Mike has extensive healthcare experience in clinical settings", 
                    "Sarah has financial experience in banking"
                ],
                'expected_top': 0  # John healthcare doc should rank first
            },
            {
                'query': 'Sarah financial background',
                'docs': [
                    "John has extensive healthcare experience in clinical settings",
                    "Mike has extensive healthcare experience in clinical settings",
                    "Sarah has financial experience in banking"  
                ],
                'expected_top': 2  # Sarah financial doc should rank first
            }
        ]
        
        for case in test_cases:
            # Create mock documents
            documents = []
            for i, content in enumerate(case['docs']):
                doc = MagicMock()
                doc.page_content = content
                doc.metadata = {}
                documents.append(doc)
            
            # Test re-ranking
            reranked = retriever_service.rerank_by_term_importance(case['query'], documents)
            
            # Validate correct attribution
            top_doc = reranked[0]
            expected_content = case['docs'][case['expected_top']]
            
            assert top_doc.page_content == expected_content, \
                f"Query '{case['query']}' should rank document with matching terms first"
            
            # Validate no cross-contamination
            top_score = top_doc.metadata['importance_score']
            other_scores = [doc.metadata['importance_score'] for doc in reranked[1:]]
            
            assert all(top_score > score for score in other_scores), \
                "Most relevant document should have highest importance score"
    
    @pytest.mark.semantic_bias
    @pytest.mark.foundation  
    def test_error_handling_graceful_fallback(self, retriever_service):
        """Test FI-05: Graceful error handling in re-ranking."""
        
        # Test with empty documents
        result = retriever_service.rerank_by_term_importance("test query", [])
        assert result == [], "Should handle empty document list"
        
        # Test with malformed documents
        bad_doc = MagicMock()
        bad_doc.page_content = None  # This could cause errors
        bad_doc.metadata = {}
        
        documents = [bad_doc]
        
        # Should not crash
        try:
            reranked = retriever_service.rerank_by_term_importance("test", documents)
            # If it doesn't crash, it should return original docs
            assert len(reranked) <= len(documents), "Should handle errors gracefully"
        except Exception:
            pytest.fail("Should not raise exceptions on malformed input")
        
        # Test with very long query
        very_long_query = "word " * 1000  # 1000 words
        normal_doc = MagicMock()
        normal_doc.page_content = "This is a normal document with some words"
        normal_doc.metadata = {}
        
        result = retriever_service.rerank_by_term_importance(very_long_query, [normal_doc])
        assert len(result) == 1, "Should handle very long queries"
        assert 'importance_score' in result[0].metadata, "Should still add importance scores" 