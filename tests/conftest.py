#!/usr/bin/env python3
"""pytest configuration and fixtures for enhanced retrieval tests"""

import pytest
import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def pytest_configure(config):
    """Configure logging for pytest runs"""
    # Load environment variables
    load_dotenv()
    
    # Enable development mode for tests to reduce API costs and improve speed
    os.environ["DEVELOPMENT_MODE"] = "true"
    os.environ["CACHE_EMBEDDINGS"] = "true"
    os.environ["BATCH_EMBEDDINGS"] = "true"
    
    # Ensure logs directory exists
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create detailed log file for each test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_log_file = os.path.join(log_dir, f"test_run_{timestamp}.log")
    
    # Configure file handler for detailed logging
    file_handler = logging.FileHandler(detailed_log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s: %(message)s (%(filename)s:%(lineno)d)',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Log test session start
    logging.info("=" * 80)
    logging.info(f"TEST SESSION STARTED: {datetime.now()}")
    logging.info(f"Detailed logs: {detailed_log_file}")
    logging.info("DEVELOPMENT MODE: Enabled (fast mock embeddings)")
    logging.info("EMBEDDING CACHE: Enabled")
    logging.info("BATCH EMBEDDINGS: Enabled")
    logging.info("=" * 80)

@pytest.fixture(scope="session")
def retriever_service():
    """Session-level retriever service for testing"""
    load_dotenv()
    
    # Ensure development mode is enabled
    os.environ["DEVELOPMENT_MODE"] = "true"
    os.environ["CACHE_EMBEDDINGS"] = "true"
    os.environ["BATCH_EMBEDDINGS"] = "true"
    
    logging.info("🔧 Initializing RetrieverService for test session")
    
    from retriever import RetrieverService
    service = RetrieverService()
    
    # Log cache statistics if available
    if hasattr(service.enhanced_retriever.embedding_function, 'get_cache_stats'):
        stats = service.enhanced_retriever.embedding_function.get_cache_stats()
        logging.info(f"📊 Initial embedding cache stats: {stats}")
    
    logging.info("✅ RetrieverService initialized successfully")
    return service

@pytest.fixture(autouse=True)
def log_test_info(request):
    """Automatically log test start/end for every test"""
    test_name = request.node.name
    start_time = datetime.now()
    logging.info(f"🧪 TEST START: {test_name}")
    
    yield
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logging.info(f"✅ TEST END: {test_name} (Duration: {duration:.2f}s)")

def pytest_runtest_logreport(report):
    """Log test results with performance information"""
    if report.when == "call":
        if report.passed:
            logging.info(f"✅ PASSED: {report.nodeid}")
        elif report.failed:
            logging.error(f"❌ FAILED: {report.nodeid}")
            if hasattr(report, 'longrepr') and report.longrepr:
                logging.error(f"Error details: {report.longrepr}")
        elif report.skipped:
            logging.warning(f"⏭️  SKIPPED: {report.nodeid}")
            if hasattr(report, 'longrepr') and report.longrepr:
                logging.warning(f"Skip reason: {report.longrepr}")

def pytest_unconfigure(config):
    """Clean up after pytest runs"""
    logging.info("=" * 80)
    logging.info(f"TEST SESSION ENDED: {datetime.now()}")
    logging.info("=" * 80) 