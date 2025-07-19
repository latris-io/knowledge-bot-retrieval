#!/bin/bash

# Foundation Improvements Test Runner
# Comprehensive testing suite for all 8 Foundation Improvements (FI-01 through FI-08)

set -e  # Exit on any error

echo "🧪 FOUNDATION IMPROVEMENTS TEST SUITE"
echo "======================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "info")
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
        "success") 
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "warning")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "error")
            echo -e "${RED}❌ $message${NC}"
            ;;
    esac
}

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_status "warning" "Virtual environment not detected. Activating venv..."
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
        print_status "success" "Virtual environment activated"
    else
        print_status "error" "Virtual environment not found. Please create with: python -m venv venv"
        exit 1
    fi
fi

# Install test dependencies if needed
print_status "info" "Checking test dependencies..."
pip install -q pytest pytest-asyncio pytest-mock pytest-timeout

echo
print_status "info" "Starting comprehensive Foundation Improvements testing..."
echo

# Test categories mapping
declare -A TEST_CATEGORIES=(
    ["foundation"]="All Foundation Improvements (FI-01 through FI-08)"
    ["retrieval_performance"]="FI-01: Enhanced Retrieval System Performance"
    ["topic_detection"]="FI-02: Semantic Topic Change Detection" 
    ["markdown"]="FI-03: Production-Grade Markdown Processing"
    ["enhanced_retrieval"]="FI-04: Content-Agnostic Enhanced Retrieval System"
    ["semantic_bias"]="FI-05: Content-Agnostic Semantic Bias Fix"
    ["hallucination"]="FI-06: LLM Hallucination Prevention"
    ["streaming"]="FI-07: Smart Streaming Enhancement"
    ["quality_improvements"]="FI-08: Enhanced Retrieval Quality Improvements"
    ["integration"]="Production Integration Tests (NO MOCKING - requires server)"
    ["unit"]="Unit Tests (with mocking for isolated testing)"
    ["production"]="Production Integration Tests (same as integration)"
    ["performance"]="Performance Tests"
)

# Function to run specific test category
run_test_category() {
    local category=$1
    local description=${TEST_CATEGORIES[$category]}
    
    print_status "info" "Testing: $description"
    echo
    
    if pytest -m "$category" -v --tb=short --maxfail=5; then
        print_status "success" "$description - PASSED"
    else
        print_status "error" "$description - FAILED"
        return 1
    fi
    
    echo
}

# Function to run all Foundation Improvement tests
run_foundation_tests() {
    print_status "info" "Running ALL Foundation Improvements Tests..."
    echo
    
    local failed_tests=()
    
    # Test each Foundation Improvement
    for category in retrieval_performance topic_detection markdown enhanced_retrieval semantic_bias hallucination streaming quality_improvements; do
        if ! run_test_category "$category"; then
            failed_tests+=("$category")
        fi
    done
    
    # Integration tests
    if ! run_test_category "integration"; then
        failed_tests+=("integration")
    fi
    
    # Summary
    echo "======================================"
    if [[ ${#failed_tests[@]} -eq 0 ]]; then
        print_status "success" "ALL FOUNDATION IMPROVEMENTS TESTS PASSED! 🎉"
        print_status "success" "All 8 Foundation Improvements are fully functional"
    else
        print_status "error" "Some tests failed: ${failed_tests[*]}"
        return 1
    fi
}

# Function to run quick validation tests
run_quick_validation() {
    print_status "info" "Running quick validation tests..."
    echo
    
    # Test just the core functionality of each FI
    pytest tests/foundation/ -k "basic or validation or core" -v --maxfail=3
    
    if [[ $? -eq 0 ]]; then
        print_status "success" "Quick validation PASSED"
    else
        print_status "error" "Quick validation FAILED"
        return 1
    fi
}

# Function to run performance tests
run_performance_tests() {
    print_status "info" "Running performance tests..."
    echo
    
    pytest -m "performance" -v --tb=short
    
    if [[ $? -eq 0 ]]; then
        print_status "success" "Performance tests PASSED"
    else
        print_status "error" "Performance tests FAILED"  
        return 1
    fi
}

# Function to show test statistics
show_test_stats() {
    print_status "info" "Generating test statistics..."
    echo
    
    # Count tests by category
    for category in "${!TEST_CATEGORIES[@]}"; do
        local count=$(pytest --collect-only -m "$category" 2>/dev/null | grep "<Function" | wc -l)
        if [[ $count -gt 0 ]]; then
            echo "  $category: $count tests"
        fi
    done
    
    echo
    local total_tests=$(pytest --collect-only tests/ 2>/dev/null | grep "<Function" | wc -l)
    print_status "info" "Total test functions: $total_tests"
}

# Main script logic
case "${1:-all}" in
    "all"|"foundation")
        run_foundation_tests
        ;;
    "quick")
        run_quick_validation
        ;;
    "performance")  
        run_performance_tests
        ;;
    "stats")
        show_test_stats
        ;;
    "fi-01"|"retrieval_performance")
        run_test_category "retrieval_performance"
        ;;
    "fi-02"|"topic_detection")
        run_test_category "topic_detection"
        ;;
    "fi-03"|"markdown")
        run_test_category "markdown"
        ;;
    "fi-04"|"enhanced_retrieval")
        run_test_category "enhanced_retrieval"
        ;;
    "fi-05"|"semantic_bias")  
        run_test_category "semantic_bias"
        ;;
    "fi-06"|"hallucination")
        run_test_category "hallucination"
        ;;
    "fi-07"|"streaming")
        run_test_category "streaming"
        ;;
    "fi-08"|"quality_improvements")
        run_test_category "quality_improvements"
        ;;
    "integration"|"production")
        print_status "warning" "Production integration tests require running server on localhost:8000"
        read -p "Is the server running? (y/N): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_test_category "integration"
        else
            print_status "error" "Please start server with: uvicorn app:app --host 0.0.0.0 --port 8000"
            exit 1
        fi
        ;;
    "unit")
        run_test_category "unit"
        ;;
    "help")
        echo "Foundation Improvements Test Runner"
        echo 
        echo "Usage: $0 [option]"
        echo
        echo "Options:"
        echo "  all, foundation     Run all Foundation Improvement tests (default)"
        echo "  quick              Run quick validation tests only"
        echo "  performance        Run performance tests only"
        echo "  stats              Show test statistics"
        echo
        echo "Individual Foundation Improvements:"
        echo "  fi-01              FI-01: Enhanced Retrieval System Performance"  
        echo "  fi-02              FI-02: Semantic Topic Change Detection"
        echo "  fi-03              FI-03: Production-Grade Markdown Processing"
        echo "  fi-04              FI-04: Content-Agnostic Enhanced Retrieval System"
        echo "  fi-05              FI-05: Content-Agnostic Semantic Bias Fix"
        echo "  fi-06              FI-06: LLM Hallucination Prevention"
        echo "  fi-07              FI-07: Smart Streaming Enhancement"
        echo "  fi-08              FI-08: Enhanced Retrieval Quality Improvements"
        echo
        echo "Test Types:"
        echo "  integration        Production Integration Tests (NO MOCKING - requires server)"
        echo "  unit              Unit Tests (with mocking for isolated testing)"
        echo "  performance        Performance tests"
        echo "  help               Show this help message"
        echo
        ;;
    *)
        print_status "error" "Unknown option: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 