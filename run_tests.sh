#!/bin/bash
# Enhanced test runner with comprehensive logging

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Enhanced Retrieval System Test Runner${NC}"
echo -e "${BLUE}==========================================${NC}"

# Create logs directory if it doesn't exist
mkdir -p tests/logs

# Get timestamp for this test run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="tests/logs/test_run_${TIMESTAMP}.log"

echo -e "${YELLOW}📝 Test logs will be written to: ${LOG_FILE}${NC}"
echo -e "${YELLOW}📊 Live logs will be displayed below${NC}"
echo ""

# Function to run specific test categories
run_tests() {
    local test_pattern="$1"
    local description="$2"
    
    echo -e "${BLUE}🔬 Running: ${description}${NC}"
    echo -e "${BLUE}Pattern: ${test_pattern}${NC}"
    echo ""
    
    # Run pytest with comprehensive logging
    pytest ${test_pattern} \
        --log-cli-level=INFO \
        --log-cli-format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s" \
        --log-cli-date-format="%H:%M:%S" \
        --log-file="$LOG_FILE" \
        --log-file-level=DEBUG \
        --log-file-format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s (%(filename)s:%(lineno)d)" \
        --log-file-date-format="%Y-%m-%d %H:%M:%S" \
        --capture=no \
        -v
    
    local exit_code=$?
    echo ""
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ ${description} - PASSED${NC}"
    else
        echo -e "${RED}❌ ${description} - FAILED${NC}"
    fi
    
    echo ""
    return $exit_code
}

# Parse command line arguments
case "${1:-all}" in
    "fast")
        echo -e "${YELLOW}🏃 Running fast tests only (no external dependencies)${NC}"
        run_tests "tests/foundation/test_enhanced_retrieval.py -k \"not multi_vector_search and not enhanced_vs_original and not hierarchical_search\"" "Fast Unit Tests"
        ;;
    "integration")
        echo -e "${YELLOW}🔗 Running integration tests (requires ChromaDB)${NC}"
        run_tests "tests/foundation/test_enhanced_retrieval.py -k \"multi_vector_search or enhanced_vs_original or hierarchical_search\"" "Integration Tests"
        ;;
    "foundation")
        echo -e "${YELLOW}🏗️ Running all Foundation Improvement tests${NC}"
        run_tests "tests/foundation/" "Foundation Improvements (FI-01 to FI-04)"
        ;;
    "verbose")
        echo -e "${YELLOW}📢 Running all tests with maximum verbosity${NC}"
        run_tests "tests/ -vvv --tb=long" "All Tests (Verbose)"
        ;;
    "debug")
        echo -e "${YELLOW}🐛 Running tests in debug mode${NC}"
        run_tests "tests/ --pdb --pdbcls=IPython.terminal.debugger:Pdb" "Debug Mode Tests"
        ;;
    "all"|"")
        echo -e "${YELLOW}🎯 Running all enhanced retrieval tests${NC}"
        run_tests "tests/foundation/test_enhanced_retrieval.py" "Enhanced Retrieval System Tests"
        ;;
    *)
        echo -e "${RED}❌ Unknown test category: $1${NC}"
        echo ""
        echo -e "${BLUE}Usage: $0 [category]${NC}"
        echo ""
        echo -e "${BLUE}Available categories:${NC}"
        echo -e "  ${GREEN}fast${NC}        - Fast unit tests (no external dependencies)"
        echo -e "  ${GREEN}integration${NC} - Integration tests (requires ChromaDB)"
        echo -e "  ${GREEN}foundation${NC}  - All Foundation Improvement tests"
        echo -e "  ${GREEN}verbose${NC}     - All tests with maximum verbosity"
        echo -e "  ${GREEN}debug${NC}       - Tests with debugger support"
        echo -e "  ${GREEN}all${NC}         - All enhanced retrieval tests (default)"
        echo ""
        exit 1
        ;;
esac

exit_code=$?

echo -e "${BLUE}==========================================${NC}"
echo -e "${YELLOW}📝 Detailed logs available in: ${LOG_FILE}${NC}"

if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests completed successfully!${NC}"
else
    echo -e "${RED}💥 Some tests failed. Check logs for details.${NC}"
fi

echo -e "${BLUE}==========================================${NC}"
exit $exit_code 