"""
FI-08: Enhanced Retrieval Quality Improvements Tests

Tests for Shannon entropy calculation, information density scoring, and smart deduplication.
"""

import pytest
from unittest.mock import MagicMock
from retriever import RetrieverService


class TestQualityImprovements:
    
    @pytest.fixture
    def retriever_service(self):
        """Create a RetrieverService instance for testing."""
        return RetrieverService()
    
    @pytest.mark.quality_improvements
    @pytest.mark.foundation
    def test_shannon_entropy_calculation(self, retriever_service):
        """Test FI-08: Shannon entropy for text quality assessment."""
        
        # High-quality informative text
        high_quality = (
            "TechCorp provides comprehensive software integration solutions for enterprise clients, "
            "specializing in API development, database optimization, and cloud migration services "
            "with proven expertise across multiple industries including healthcare and finance."
        )
        
        # Low-quality repetitive text
        low_quality = "test test test test test test test test test test test test test test"
        
        # Very short text
        short_text = "Hi there"
        
        # Calculate entropy
        high_entropy = retriever_service.calculate_shannon_entropy(high_quality)
        low_entropy = retriever_service.calculate_shannon_entropy(low_quality)
        short_entropy = retriever_service.calculate_shannon_entropy(short_text)
        
        # Validate entropy scoring
        assert high_entropy > low_entropy, "Diverse text should have higher entropy than repetitive text"
        assert high_entropy > 3.0, f"High-quality text should exceed threshold: {high_entropy}"
        assert low_entropy < 3.0, f"Low-quality text should be below threshold: {low_entropy}" 
        assert short_entropy == 0.0, "Very short text should have zero entropy"
        
        # Test edge cases
        empty_entropy = retriever_service.calculate_shannon_entropy("")
        assert empty_entropy == 0.0, "Empty text should have zero entropy"
        
        none_entropy = retriever_service.calculate_shannon_entropy(None)
        assert none_entropy == 0.0, "None input should have zero entropy"
    
    @pytest.mark.quality_improvements
    @pytest.mark.foundation
    def test_information_density_calculation(self, retriever_service):
        """Test FI-08: Information density scoring metrics."""
        
        # High information density text
        dense_text = (
            "Microservices architecture enables scalable distributed systems through containerization, "
            "API gateways, service discovery mechanisms, and fault-tolerant messaging patterns "
            "implemented via Kubernetes orchestration and circuit breaker methodologies."
        )
        
        # Low information density text  
        sparse_text = "This is a simple text with basic words and no complex concepts or detailed information."
        
        # Very repetitive text
        repetitive_text = "the the the and and and is is is for for for"
        
        # Calculate density scores
        dense_score = retriever_service.calculate_information_density(dense_text)
        sparse_score = retriever_service.calculate_information_density(sparse_text)
        repetitive_score = retriever_service.calculate_information_density(repetitive_text)
        
        # Validate density scoring
        assert dense_score > sparse_score, "Technical text should have higher density"
        assert dense_score > 0.3, f"High-quality text should exceed density threshold: {dense_score}"
        assert repetitive_score < 0.3, f"Repetitive text should be below threshold: {repetitive_score}"
        
        # Test components of density calculation
        # Word diversity component
        diverse_words = "advanced machine learning algorithms utilize neural networks"
        repetitive_words = "machine machine machine learning learning learning"
        
        diverse_density = retriever_service.calculate_information_density(diverse_words)
        repetitive_density = retriever_service.calculate_information_density(repetitive_words)
        
        assert diverse_density > repetitive_density, "Diverse vocabulary should increase density"
    
    @pytest.mark.quality_improvements
    @pytest.mark.foundation  
    def test_quality_filtering_thresholds(self, retriever_service):
        """Test FI-08: Quality filtering with entropy and density thresholds."""
        
        # Create documents with varying quality levels
        high_quality_doc = MagicMock()
        high_quality_doc.page_content = (
            "TechCorp's enterprise software integration platform delivers comprehensive API management, "
            "microservices orchestration, and distributed system monitoring capabilities. The platform "
            "supports multi-cloud deployment architectures with advanced security protocols and "
            "real-time analytics for performance optimization across complex enterprise environments."
        )
        high_quality_doc.metadata = {}
        
        medium_quality_doc = MagicMock()
        medium_quality_doc.page_content = (
            "TechCorp provides software solutions for business clients. The company offers "
            "various services including system integration and technical support for enterprises."
        )
        medium_quality_doc.metadata = {}
        
        low_quality_doc = MagicMock()
        low_quality_doc.page_content = "test test test test test test test test test"
        low_quality_doc.metadata = {}
        
        very_short_doc = MagicMock()
        very_short_doc.page_content = "Hi"
        very_short_doc.metadata = {}
        
        documents = [low_quality_doc, high_quality_doc, very_short_doc, medium_quality_doc]
        
        # Apply quality filtering
        filtered = retriever_service.filter_by_quality(documents)
        
        # Validate filtering results
        assert len(filtered) < len(documents), "Should filter out some low-quality documents"
        assert high_quality_doc in filtered, "High-quality document should pass filter"
        assert very_short_doc not in filtered, "Very short document should be filtered out"
        assert low_quality_doc not in filtered, "Low-quality repetitive document should be filtered out"
        
        # Check quality metrics were added
        for doc in filtered:
            assert 'shannon_entropy' in doc.metadata, "Should add entropy scores"
            assert 'information_density' in doc.metadata, "Should add density scores" 
            assert 'quality_score' in doc.metadata, "Should add combined quality score"
            
            # Validate quality thresholds
            assert doc.metadata['shannon_entropy'] >= 3.0, "Filtered docs should meet entropy threshold"
            assert doc.metadata['information_density'] >= 0.3, "Filtered docs should meet density threshold"
    
    @pytest.mark.quality_improvements
    @pytest.mark.foundation
    def test_smart_deduplication_similarity(self, retriever_service):
        """Test FI-08: Smart deduplication based on content similarity."""
        
        # Create documents with varying similarity levels
        original_doc = MagicMock()
        original_doc.page_content = "TechCorp specializes in enterprise software integration solutions"
        original_doc.metadata = {}
        
        very_similar_doc = MagicMock()
        very_similar_doc.page_content = "TechCorp specializes in enterprise software integration services" 
        very_similar_doc.metadata = {}
        
        somewhat_similar_doc = MagicMock()
        somewhat_similar_doc.page_content = "TechCorp provides business software solutions for enterprises"
        somewhat_similar_doc.metadata = {}
        
        different_doc = MagicMock() 
        different_doc.page_content = "DataSys offers data processing and analytics services"
        different_doc.metadata = {}
        
        documents = [original_doc, very_similar_doc, somewhat_similar_doc, different_doc]
        
        # Test deduplication with default threshold (0.85)
        deduplicated = retriever_service.smart_deduplicate(documents)
        
        # Validate deduplication
        assert len(deduplicated) < len(documents), "Should remove similar duplicates"
        assert original_doc in deduplicated, "Should keep original document"
        assert different_doc in deduplicated, "Should keep different document" 
        assert very_similar_doc not in deduplicated, "Should remove very similar document"
        
        # Check similarity scores were added
        for doc in deduplicated:
            assert 'dedup_similarity' in doc.metadata, "Should add similarity scores"
        
        # Test with different threshold
        lenient_deduplicated = retriever_service.smart_deduplicate(documents, similarity_threshold=0.95)
        
        # More lenient threshold should keep more documents
        assert len(lenient_deduplicated) >= len(deduplicated), "Lenient threshold should keep more docs"
    
    @pytest.mark.quality_improvements
    @pytest.mark.foundation
    def test_complete_quality_enhancement_pipeline(self, retriever_service):
        """Test FI-08: Complete quality enhancement pipeline integration."""
        
        query = "TechCorp software integration"
        
        # Create diverse document set for comprehensive testing
        documents = []
        
        # High-quality relevant document
        high_qual = MagicMock()
        high_qual.page_content = (
            "TechCorp's software integration platform provides comprehensive API management, "
            "microservices orchestration, and enterprise system connectivity solutions with "
            "advanced security protocols and real-time monitoring capabilities for optimal performance."
        )
        high_qual.metadata = {}
        documents.append(high_qual)
        
        # Duplicate document (should be removed)
        duplicate = MagicMock() 
        duplicate.page_content = (
            "TechCorp's software integration platform provides comprehensive API management, "
            "microservices orchestration, and enterprise system connectivity solutions with "
            "advanced security protocols and real-time monitoring capabilities for performance."
        )  # Very similar to above
        duplicate.metadata = {}
        documents.append(duplicate)
        
        # Medium quality document
        medium_qual = MagicMock()
        medium_qual.page_content = "TechCorp offers software integration services for business clients"
        medium_qual.metadata = {}
        documents.append(medium_qual)
        
        # Low quality document (should be filtered out)
        low_qual = MagicMock()
        low_qual.page_content = "software software software integration integration integration"
        low_qual.metadata = {}
        documents.append(low_qual)
        
        # Irrelevant high-quality document (should rank lower)
        irrelevant = MagicMock()
        irrelevant.page_content = (
            "DataSys provides comprehensive data analytics and machine learning solutions "
            "with advanced statistical modeling and predictive analytics capabilities for "
            "enterprise business intelligence and decision support systems."
        )
        irrelevant.metadata = {}
        documents.append(irrelevant)
        
        # Apply complete quality enhancements
        enhanced = retriever_service.apply_quality_enhancements(query, documents)
        
        # Validate complete pipeline
        assert len(enhanced) < len(documents), "Should filter and deduplicate"
        assert low_qual not in enhanced, "Should filter low-quality document"
        assert duplicate not in enhanced, "Should deduplicate similar document"
        assert high_qual in enhanced, "Should keep high-quality relevant document"
        
        # Check all enhancements were applied
        for doc in enhanced:
            assert 'shannon_entropy' in doc.metadata, "Should have quality filtering metadata"
            assert 'dedup_similarity' in doc.metadata, "Should have deduplication metadata"  
            assert 'importance_score' in doc.metadata, "Should have re-ranking metadata"
            assert 'final_rank' in doc.metadata, "Should have final ranking"
            assert 'enhancement_applied' in doc.metadata, "Should mark enhancements applied"
        
        # Validate ranking order
        rankings = [doc.metadata['final_rank'] for doc in enhanced]
        assert rankings == sorted(rankings), "Final rankings should be ordered"
        
        # Most relevant document should rank first
        top_doc = enhanced[0] 
        assert 'TechCorp' in top_doc.page_content, "Most relevant document should rank first"
        assert 'software integration' in top_doc.page_content.lower(), "Should match query terms"
    
    @pytest.mark.quality_improvements
    @pytest.mark.foundation
    def test_quality_enhancement_error_handling(self, retriever_service):
        """Test FI-08: Error handling in quality enhancement pipeline."""
        
        # Test empty document list
        result = retriever_service.apply_quality_enhancements("test", [])
        assert result == [], "Should handle empty document list"
        
        # Test with malformed documents
        bad_doc = MagicMock()
        bad_doc.page_content = None  # Could cause errors
        bad_doc.metadata = {}
        
        try:
            result = retriever_service.apply_quality_enhancements("test", [bad_doc])
            # Should not crash
            assert isinstance(result, list), "Should return list even with bad input"
        except Exception:
            pytest.fail("Should handle malformed documents gracefully")
        
        # Test each component's error handling
        assert retriever_service.filter_by_quality([]) == [], "Quality filter should handle empty list"
        assert retriever_service.smart_deduplicate([]) == [], "Deduplication should handle empty list"
        assert retriever_service.rerank_by_term_importance("test", []) == [], "Re-ranking should handle empty list"
    
    @pytest.mark.quality_improvements
    @pytest.mark.foundation
    @pytest.mark.integration
    def test_quality_improvements_performance(self, retriever_service):
        """Test FI-08: Performance characteristics of quality improvements."""
        
        import time
        
        # Create larger document set for performance testing
        documents = []
        for i in range(20):
            doc = MagicMock()
            doc.page_content = f"TechCorp document {i} contains software integration information with detailed technical specifications and comprehensive implementation guidelines for enterprise clients."
            doc.metadata = {}
            documents.append(doc)
        
        # Measure enhancement performance
        start_time = time.time()
        enhanced = retriever_service.apply_quality_enhancements("TechCorp software", documents)
        end_time = time.time()
        
        enhancement_time = end_time - start_time
        
        # Performance validation
        assert enhancement_time < 5.0, f"Quality enhancements should be fast: {enhancement_time}s"
        assert len(enhanced) > 0, "Should return enhanced results"
        assert len(enhanced) <= len(documents), "Should not add documents"
        
        # Validate quality was actually improved
        quality_scores = [doc.metadata.get('quality_score', 0) for doc in enhanced]
        assert all(score > 0 for score in quality_scores), "All results should have positive quality scores"
        
        # Check ordering
        importance_scores = [doc.metadata.get('importance_score', 0) for doc in enhanced]
        # Should be in descending order (highest importance first)
        for i in range(len(importance_scores) - 1):
            assert importance_scores[i] >= importance_scores[i + 1], "Should be ranked by importance" 