#!/usr/bin/env python3
"""
Multi-Tenant Security Tests
Validates that company_id/bot_id isolation prevents cross-tenant data leaks
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch
from app import ask_question, get_session_history, validate_tenant_access

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMultiTenantSecurity:
    """Test multi-tenant security isolation"""

    def test_session_history_isolation(self):
        """Test that session histories are isolated by company_id/bot_id/session_id"""
        
        # Same session_id but different company/bot combinations
        session_id = "test_session_123"
        
        # Company 1, Bot 1
        history_1_1 = get_session_history(company_id=1, bot_id=1, session_id=session_id)
        history_1_1.add_user_message("Company 1 Bot 1 message")
        
        # Company 1, Bot 2 (different bot, same company)
        history_1_2 = get_session_history(company_id=1, bot_id=2, session_id=session_id)
        history_1_2.add_user_message("Company 1 Bot 2 message")
        
        # Company 2, Bot 1 (different company)
        history_2_1 = get_session_history(company_id=2, bot_id=1, session_id=session_id)
        history_2_1.add_user_message("Company 2 Bot 1 message")
        
        # Verify complete isolation
        assert len(history_1_1.messages) == 1
        assert len(history_1_2.messages) == 1
        assert len(history_2_1.messages) == 1
        
        # Verify content isolation
        assert "Company 1 Bot 1" in str(history_1_1.messages[0])
        assert "Company 1 Bot 2" in str(history_1_2.messages[0])
        assert "Company 2 Bot 1" in str(history_2_1.messages[0])
        
        # Verify no cross-contamination
        assert "Company 1 Bot 2" not in str(history_1_1.messages[0])
        assert "Company 2 Bot 1" not in str(history_1_1.messages[0])
        
        logger.info("✅ Session history isolation test passed")

    def test_tenant_access_validation(self):
        """Test that tenant access validation blocks unauthorized access"""
        
        # Test case 1: No documents found - should block access
        assert validate_tenant_access(company_id=7, bot_id=6, documents_found=0) == False
        
        # Test case 2: Documents found - should allow access
        assert validate_tenant_access(company_id=7, bot_id=5, documents_found=4) == True
        
        # Test case 3: Edge case - zero documents always blocks
        assert validate_tenant_access(company_id=999, bot_id=999, documents_found=0) == False
        
        logger.info("✅ Tenant access validation test passed")

    @pytest.mark.asyncio
    async def test_security_response_no_fallbacks(self):
        """Test that security response is returned when no documents found, with no fallbacks"""
        
        # Mock the retriever to return no documents (simulating company_id/bot_id mismatch)
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.return_value = []
        mock_retriever._get_relevant_documents.return_value = []
        
        # Mock the retriever service
        with patch('app.RetrieverService') as mock_service:
            mock_service.return_value.build_enhanced_retriever.return_value = mock_retriever
            mock_service.return_value.embedding_function = Mock()
            
            try:
                # Test query that should trigger security block
                result = await ask_question(
                    question="when is brentwood open?",
                    company_id=7,
                    bot_id=6,  # Mismatched bot_id (data has bot_id=5)
                    session_id="security_test_session",
                    streaming=False,
                    verbose=True
                )
                
                # Verify security response
                assert "Access Restricted" in result["result"]
                assert "Company ID: 7" in result["result"]
                assert "Bot ID: 6" in result["result"]
                assert "Authorized documents: 0" in result["result"]
                assert "For security reasons" in result["result"]
                
                # Verify no source documents
                assert result["source_documents"] == []
                
                # Verify no fallback information (the critical security check)
                assert "7:00 AM - 4:00 PM" not in result["result"]  # Brentwood hours
                assert "7004 Moores Lane" not in result["result"]   # Brentwood address
                assert "629-260-7397" not in result["result"]      # Phone numbers
                assert "previous conversation context" not in result["result"]
                
                logger.info("✅ Security response test passed - no fallbacks allowed")
                
            except Exception as e:
                logger.error(f"❌ Security test failed: {e}")
                raise

    @pytest.mark.asyncio 
    async def test_cross_bot_isolation_within_company(self):
        """Test that bots within the same company cannot access each other's data"""
        
        # Simulate Bot 5 having documents, Bot 6 having none
        def mock_retriever_factory(company_id, bot_id, **kwargs):
            mock_retriever = Mock()
            if company_id == 7 and bot_id == 5:
                # Bot 5 has documents
                mock_doc = Mock()
                mock_doc.page_content = "Brentwood office hours: Monday-Thursday 7:00 AM - 4:00 PM"
                mock_doc.metadata = {"file_name": "Lucas Offices.xlsx", "bot_id": 5}
                mock_retriever.get_relevant_documents.return_value = [mock_doc]
                mock_retriever._get_relevant_documents.return_value = [mock_doc]
            else:
                # Bot 6 has no documents
                mock_retriever.get_relevant_documents.return_value = []
                mock_retriever._get_relevant_documents.return_value = []
            return mock_retriever
        
        with patch('app.RetrieverService') as mock_service:
            mock_service.return_value.build_enhanced_retriever.side_effect = mock_retriever_factory
            mock_service.return_value.embedding_function = Mock()
            
            # Test Bot 5 (has documents) - should get normal response
            result_bot5 = await ask_question(
                question="when is brentwood open?",
                company_id=7,
                bot_id=5,
                session_id="bot5_session",
                streaming=False,
                verbose=True
            )
            
            # Test Bot 6 (no documents) - should get security block
            result_bot6 = await ask_question(
                question="when is brentwood open?", 
                company_id=7,
                bot_id=6,
                session_id="bot6_session",
                streaming=False,
                verbose=True
            )
            
            # Verify Bot 5 gets data (has authorization)
            assert "Access Restricted" not in result_bot5["result"]
            
            # Verify Bot 6 gets security block (no authorization)
            assert "Access Restricted" in result_bot6["result"]
            assert "Bot ID: 6" in result_bot6["result"]
            assert "Authorized documents: 0" in result_bot6["result"]
            
            # Critical: Verify Bot 6 cannot access Bot 5's Brentwood data
            assert "7:00 AM - 4:00 PM" not in result_bot6["result"]
            assert "Brentwood" not in result_bot6["result"]
            
            logger.info("✅ Cross-bot isolation test passed")

    def test_session_key_format(self):
        """Test that session keys include all tenant isolation parameters"""
        
        # Test different combinations produce different session histories
        combinations = [
            (1, 1, "session123"),
            (1, 2, "session123"),  # Same company, different bot
            (2, 1, "session123"),  # Different company, same bot
            (1, 1, "session456"),  # Same company/bot, different session
        ]
        
        histories = []
        for company_id, bot_id, session_id in combinations:
            history = get_session_history(company_id, bot_id, session_id)
            history.add_user_message(f"Test message {company_id}-{bot_id}-{session_id}")
            histories.append(history)
        
        # Verify all histories are completely separate
        for i, history in enumerate(histories):
            assert len(history.messages) == 1
            expected_content = f"Test message {combinations[i][0]}-{combinations[i][1]}-{combinations[i][2]}"
            assert expected_content in str(history.messages[0])
            
            # Verify no cross-contamination with other histories
            for j, other_history in enumerate(histories):
                if i != j:
                    other_content = f"Test message {combinations[j][0]}-{combinations[j][1]}-{combinations[j][2]}"
                    assert other_content not in str(history.messages[0])
        
        logger.info("✅ Session key format test passed")

if __name__ == "__main__":
    # Run security tests
    test_suite = TestMultiTenantSecurity()
    
    # Run synchronous tests
    test_suite.test_session_history_isolation()
    test_suite.test_tenant_access_validation()
    test_suite.test_session_key_format()
    
    # Run async tests
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_suite.test_security_response_no_fallbacks())
    loop.run_until_complete(test_suite.test_cross_bot_isolation_within_company())
    
    print("🔒 All multi-tenant security tests passed!") 