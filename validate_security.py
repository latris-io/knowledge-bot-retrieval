#!/usr/bin/env python3
"""
Quick Security Validation Script - NO MOCKING
Tests real multi-tenant isolation with actual system components
"""

import sys
import time
import logging

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)

def main():
    print("🔒 VALIDATING MULTI-TENANT SECURITY (REAL COMPONENTS)")
    print("=" * 60)
    
    try:
        from app import get_session_history, validate_tenant_access
        
        # Test 1: Session History Isolation
        print("\n📋 Test 1: Session History Isolation")
        session_id = f'security_test_{int(time.time())}'
        
        # Create histories for different tenants with same session ID
        history_1_1 = get_session_history(company_id=1, bot_id=1, session_id=session_id)
        history_1_2 = get_session_history(company_id=1, bot_id=2, session_id=session_id)  # Different bot
        history_2_1 = get_session_history(company_id=2, bot_id=1, session_id=session_id)  # Different company
        
        # Add tenant-specific messages
        history_1_1.add_user_message("Confidential data for Company 1 Bot 1")
        history_1_2.add_user_message("Confidential data for Company 1 Bot 2") 
        history_2_1.add_user_message("Confidential data for Company 2 Bot 1")
        
        # Validate complete isolation
        isolation_check = (
            history_1_1 is not history_1_2 and  # Different objects
            history_1_2 is not history_2_1 and
            history_1_1 is not history_2_1 and
            len(history_1_1.messages) == 1 and  # Each has only its own message
            len(history_1_2.messages) == 1 and
            len(history_2_1.messages) == 1
        )
        
        print(f"   ✅ Session objects isolated: {isolation_check}")
        
        # Validate no cross-contamination
        h1_content = str(history_1_1.messages[0])
        h2_content = str(history_1_2.messages[0])
        h3_content = str(history_2_1.messages[0])
        
        no_contamination = (
            "Company 1 Bot 1" in h1_content and
            "Company 1 Bot 2" not in h1_content and  # No cross-bot leak
            "Company 2 Bot 1" not in h1_content      # No cross-company leak
        )
        
        print(f"   ✅ No cross-tenant contamination: {no_contamination}")
        
        # Test 2: Tenant Access Validation
        print("\n📋 Test 2: Tenant Access Validation")
        
        # Test cases: (company_id, bot_id, doc_count, expected_access)
        test_cases = [
            (7, 6, 0, False),    # No documents - should block
            (7, 5, 4, True),     # Has documents - should allow
            (999, 999, 0, False) # Invalid tenant - should block
        ]
        
        all_validation_passed = True
        for company_id, bot_id, doc_count, expected in test_cases:
            result = validate_tenant_access(company_id, bot_id, doc_count)
            passed = result == expected
            status = "✅" if passed else "❌"
            print(f"   {status} Company {company_id}, Bot {bot_id}, Docs {doc_count}: {'Allow' if result else 'Block'}")
            if not passed:
                all_validation_passed = False
        
        print(f"   ✅ All validation tests passed: {all_validation_passed}")
        
        # Test 3: Security Logging
        print("\n📋 Test 3: Security Logging")
        
        import io
        import logging
        
        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        
        app_logger = logging.getLogger("app")
        app_logger.addHandler(handler)
        
        # Trigger security validation (should log warning)
        validate_tenant_access(company_id=7, bot_id=6, documents_found=0)
        
        # Check log output
        log_output = log_capture.getvalue()
        security_logged = (
            "SECURITY" in log_output and
            "company_id=7" in log_output and
            "bot_id=6" in log_output and
            "Blocking all fallbacks" in log_output
        )
        
        print(f"   ✅ Security violations logged: {security_logged}")
        
        # Cleanup
        app_logger.removeHandler(handler)
        
        # Final Results
        print("\n🎯 SECURITY VALIDATION RESULTS")
        print("-" * 40)
        
        all_tests_passed = isolation_check and no_contamination and all_validation_passed and security_logged
        
        if all_tests_passed:
            print("✅ ALL SECURITY TESTS PASSED")
            print("🔒 Multi-tenant isolation is SECURE")
            print("🚫 Cross-tenant data leaks PREVENTED")
            print("📝 Security violations properly LOGGED")
            return 0
        else:
            print("❌ SECURITY TESTS FAILED")
            print("⚠️  POTENTIAL SECURITY VULNERABILITY")
            return 1
            
    except ImportError as e:
        print(f"❌ Cannot import security functions: {e}")
        print("⚠️  Run from project root with proper environment")
        return 1
    except Exception as e:
        print(f"❌ Security validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 