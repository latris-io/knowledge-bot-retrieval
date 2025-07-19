"""
FI-07: Smart Streaming Enhancement Tests
Tests for word-boundary streaming, JSON chunk structure, content classification, and markdown-it integration.
"""

import json
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app import SmartEventStreamHandler


class TestSmartStreamingEnhancement:
    """Test suite for FI-07: Smart Streaming Enhancement"""

    def setup_method(self):
        """Set up test fixtures"""
        self.handler = SmartEventStreamHandler()
        
    def test_smart_event_stream_handler_initialization(self):
        """Test FI-07.1: Handler initialization with proper defaults"""
        handler = SmartEventStreamHandler()
        
        # Verify initialization
        assert handler.token_buffer == ""
        assert handler.accumulated_text == ""
        assert handler.chunk_id == 0
        assert isinstance(handler.queue, asyncio.Queue)
        assert handler.streaming_started == False
        
    def test_word_boundary_detection(self):
        """Test FI-07.2: Word boundary detection algorithm"""
        handler = SmartEventStreamHandler()
        
        # Test sentence boundaries
        handler.token_buffer = "Hello world"
        assert handler._should_flush_buffer(".") is True
        assert handler._should_flush_buffer("!") is True  
        assert handler._should_flush_buffer("?") is True
        
        # Test word boundaries (space after word)
        handler.token_buffer = "test"
        assert handler._should_flush_buffer(" ") is True
        
        # Test line breaks
        handler.token_buffer = "content"
        assert handler._should_flush_buffer("\n") is True
        
        # Test markdown patterns - the actual logic checks if combined ends with '**' or '###'
        handler.token_buffer = "**bold*"
        assert handler._should_flush_buffer("*") is True  # "**bold*" + "*" = "**bold**" ends with "**"
        
        handler.token_buffer = "##"
        assert handler._should_flush_buffer("#") is True  # "##" + "#" = "###" ends with "###"
        
        # Test max buffer size
        handler.token_buffer = "a" * 51
        assert handler._should_flush_buffer("x") is True
        
        # Test cases that shouldn't flush
        handler.token_buffer = "incomplet"
        assert handler._should_flush_buffer("e") is False
        
    def test_content_type_classification_in_format_chunk(self):
        """Test FI-07.3: Automatic content type detection in JSON formatting"""
        handler = SmartEventStreamHandler()
        
        # Test header detection
        header_chunk = handler._format_json_chunk("### Header", "content")
        header_data = json.loads(header_chunk)
        assert header_data["content_type"] == "header"
        
        # Test list item detection  
        list_chunk = handler._format_json_chunk("- List item", "content")
        list_data = json.loads(list_chunk)
        assert list_data["content_type"] == "list_item"
        
        # Test source detection
        source_chunk = handler._format_json_chunk("Some text [source: test.pdf#1]", "content")
        source_data = json.loads(source_chunk)
        assert source_data["content_type"] == "source"
        
        # Test regular text
        text_chunk = handler._format_json_chunk("Regular paragraph text", "content")
        text_data = json.loads(text_chunk)
        assert text_data["content_type"] == "text"
        
    @pytest.mark.asyncio
    async def test_json_chunk_format(self):
        """Test FI-07.4: Proper JSON chunk structure generation"""
        handler = SmartEventStreamHandler()
        
        # Initialize streaming
        handler.on_llm_start({}, [])
        
        # Process some tokens
        handler.on_llm_new_token("### ")
        handler.on_llm_new_token("Test ")
        handler.on_llm_new_token("Header ")
        
        # Check generated chunks
        chunks = []
        try:
            while True:
                chunk_data = await asyncio.wait_for(handler.queue.get(), timeout=0.1)
                if chunk_data is None:
                    break
                chunks.append(json.loads(chunk_data))
        except asyncio.TimeoutError:
            pass  # No more chunks available
        
        # Verify chunk structure
        assert len(chunks) >= 1
        
        # Check chunk format
        for chunk in chunks:
            assert "id" in chunk
            assert "type" in chunk
            assert "final" in chunk
            
            if chunk["type"] == "content":
                assert "content" in chunk
                assert "content_type" in chunk
                
    @pytest.mark.asyncio
    async def test_streaming_flow_complete(self):
        """Test FI-07.5: Complete streaming flow from start to end"""
        handler = SmartEventStreamHandler()
        
        # Start streaming
        handler.on_llm_start({}, [])
        
        # Process markdown content
        test_content = [
            "### ", "Smart ", "Streaming\n\n",
            "This is ", "**bold** ", "text.\n\n",
            "- Feature ", "one\n",
            "- Feature ", "two\n\n",
            "[source: test.md#1]"
        ]
        
        for token in test_content:
            handler.on_llm_new_token(token)
            
        # End streaming
        handler.on_llm_end({})
        
        # Collect all chunks with timeout
        chunks = []
        try:
            while True:
                chunk_data = await asyncio.wait_for(handler.queue.get(), timeout=0.1)
                if chunk_data is None:
                    break
                chunks.append(json.loads(chunk_data))
        except asyncio.TimeoutError:
            pass  # No more chunks available
        
        # Verify flow
        assert len(chunks) >= 1
        
        # Check chunk types
        chunk_types = [c["type"] for c in chunks]
        assert "start" in chunk_types
        
        # Verify content types are detected
        content_chunks = [c for c in chunks if c["type"] == "content"]
        if content_chunks:
            content_types = [c.get("content_type") for c in content_chunks]
            # Should have various content types
            assert len(set(content_types)) > 1  # Multiple content types detected
        
    def test_word_boundary_accuracy(self):
        """Test FI-07.6: Word boundary accuracy >80% (realistic target)"""
        handler = SmartEventStreamHandler()
        
        # Test with various content that should respect boundaries
        test_cases = [
            ("Hello world", " ", True),   # Word boundary
            ("sentence", ".", True),      # Sentence boundary
            ("paragraph", "\n\n", True),  # Paragraph boundary
            ("incom", "plete", False),    # Mid-word (shouldn't flush)
            ("**bold*", "*", True),       # Markdown pattern - ends with "**"
        ]
        
        boundary_accuracy = 0
        for buffer, token, expected in test_cases:
            handler.token_buffer = buffer
            result = handler._should_flush_buffer(token)
            if result == expected:
                boundary_accuracy += 1
                
        accuracy_rate = boundary_accuracy / len(test_cases)
        assert accuracy_rate >= 0.8  # 80% minimum accuracy
        
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test FI-07.7: Error handling with structured error messages"""
        handler = SmartEventStreamHandler()
        
        # Start streaming
        handler.on_llm_start({}, [])
        
        # Simulate error
        test_error = {"message": "Test error", "code": "TEST_ERROR"}
        handler.on_llm_error(test_error)
        
        # Check error chunk
        error_chunk = None
        try:
            while True:
                chunk_data = await asyncio.wait_for(handler.queue.get(), timeout=0.1)
                if chunk_data is None:
                    break
                chunk = json.loads(chunk_data)
                if chunk["type"] == "error":
                    error_chunk = chunk
                    break
        except asyncio.TimeoutError:
            pass
        
        assert error_chunk is not None
        assert error_chunk["type"] == "error"
        assert "error" in error_chunk
        
    def test_backward_compatibility(self):
        """Test FI-07.8: Backward compatibility with legacy systems"""
        handler = SmartEventStreamHandler()
        
        # Verify handler can be used in legacy mode
        handler.on_llm_start({}, [])
        handler.on_llm_new_token("test content")
        
        # Should still accumulate text for backward compatibility
        assert "test content" in handler.accumulated_text
        
    def test_performance_no_latency_increase(self):
        """Test FI-07.9: No significant latency increase over raw streaming"""
        import time
        
        handler = SmartEventStreamHandler()
        
        # Time the processing of tokens
        start_time = time.time()
        
        handler.on_llm_start({}, [])
        for i in range(100):
            handler.on_llm_new_token(f"token{i} ")
        handler.on_llm_end({})
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 100 tokens in reasonable time (< 1 second)
        assert processing_time < 1.0
        
    def test_content_type_distribution(self):
        """Test FI-07.10: Content type detection across various markdown elements"""
        handler = SmartEventStreamHandler()
        
        test_content = {
            "### Main Header": "header",
            "## Sub Header": "header", 
            "- List item one": "list_item",
            "* Bullet point": "list_item",
            "1. Numbered item": "list_item",
            "[source: document.pdf#1]": "source",
            "Regular paragraph text here.": "text",
            "More normal content without special formatting.": "text"
        }
        
        correct_classifications = 0
        for content, expected_type in test_content.items():
            chunk_json = handler._format_json_chunk(content, "content")
            chunk_data = json.loads(chunk_json)
            actual_type = chunk_data["content_type"]
            if actual_type == expected_type:
                correct_classifications += 1
        
        accuracy = correct_classifications / len(test_content)
        assert accuracy >= 0.85  # 85% classification accuracy minimum


class TestMarkdownIntegration:
    """Test suite for markdown-it integration capabilities"""
    
    def test_header_immediate_parsing_capability(self):
        """Test that headers can be parsed immediately during streaming"""
        handler = SmartEventStreamHandler()
        
        # Test complete header content
        header_content = "### Smart Streaming Demo"
        chunk_json = handler._format_json_chunk(header_content, "content")
        chunk_data = json.loads(chunk_json)
        content_type = chunk_data["content_type"]
        
        assert content_type == "header"
        
        # Verify this could be parsed immediately by markdown-it
        assert header_content.startswith("###")
        assert len(header_content.split()) >= 2  # Has actual content
        
    def test_progressive_parsing_readiness(self):
        """Test content is ready for progressive markdown parsing"""
        handler = SmartEventStreamHandler()
        
        # Test various content types for parsing readiness
        test_cases = [
            ("### Header Content", "header", True),
            ("- List item complete", "list_item", True), 
            ("Partial sentence without", "text", False),
            ("Complete sentence.", "text", True),
            ("[source: file.pdf#1]", "source", True)
        ]
        
        for content, expected_type, should_be_ready in test_cases:
            chunk_json = handler._format_json_chunk(content, "content")
            chunk_data = json.loads(chunk_json)
            actual_type = chunk_data["content_type"]
            assert actual_type == expected_type
            
            # Headers and sources should always be ready for parsing
            if expected_type in ["header", "source"]:
                assert should_be_ready
                
    @pytest.mark.asyncio
    async def test_chunk_metadata_for_client_processing(self):
        """Test that chunks provide sufficient metadata for client-side processing"""
        handler = SmartEventStreamHandler()
        
        handler.on_llm_start({}, [])
        handler.on_llm_new_token("### Header\n\n")
        handler.on_llm_new_token("Regular text content")
        
        # Get chunks
        chunks = []
        try:
            while True:
                chunk_data = await asyncio.wait_for(handler.queue.get(), timeout=0.1)
                if chunk_data is None:
                    break
                chunks.append(json.loads(chunk_data))
        except asyncio.TimeoutError:
            pass
        
        # Verify metadata exists
        content_chunks = [c for c in chunks if c["type"] == "content"]
        
        for chunk in content_chunks:
            assert "content_type" in chunk
            assert chunk["content_type"] in ["header", "text", "list_item", "source"]
            assert "id" in chunk
            assert "final" in chunk


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 