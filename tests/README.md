# Testing Guide - Enhanced Retrieval System

## 🧪 Test Logging Setup

This testing suite provides comprehensive logging capabilities to help debug and verify system behavior during test execution.

### 📋 What's Configured

#### 1. **pytest.ini Configuration**
- **Live Console Logs**: Real-time log output during tests (`--log-cli-level=INFO`)
- **Detailed File Logs**: Complete debug logs written to timestamped files (`--log-file-level=DEBUG`)
- **Formatted Output**: Clean, readable log formatting with timestamps and source locations
- **No Capture**: Immediate output display (`--capture=no`)

#### 2. **conftest.py Setup**
- **Automatic Test Tracking**: Each test start/end is logged with emoji indicators
- **Session Management**: Logs for test session initialization and cleanup
- **Timestamped Log Files**: Each test run gets its own log file with timestamp
- **Comprehensive Coverage**: Logs test results (pass/fail/skip) with details

#### 3. **Enhanced Test Runner (`run_tests.sh`)**
- **Categorized Testing**: Different test categories (fast, integration, verbose, debug)
- **Colored Output**: Clear visual indicators for different test states
- **Log File Management**: Automatic log file creation and organization
- **Usage Guide**: Built-in help and usage examples

### 📊 Log Files Generated

When you run tests, the following log files are created in `tests/logs/`:

1. **`pytest.log`** - Main pytest log file (configured in pytest.ini)
2. **`test_run_YYYYMMDD_HHMMSS.log`** - Detailed timestamped log for each test run
3. **Individual test logs** - Comprehensive details for debugging

### 🚀 How to Use

#### Quick Test Run
```bash
# Run all enhanced retrieval tests with full logging
./run_tests.sh

# Run specific test categories
./run_tests.sh fast        # Fast tests only (no external dependencies)
./run_tests.sh integration # Integration tests (requires ChromaDB)
./run_tests.sh verbose     # Maximum verbosity
./run_tests.sh debug       # Debug mode with breakpoints
```

#### Direct pytest Usage
```bash
# Run with live logging
pytest tests/foundation/test_enhanced_retrieval.py -v --log-cli-level=INFO

# Run with file logging only
pytest tests/foundation/test_enhanced_retrieval.py --log-file=tests/logs/my_test.log

# Run specific test with debug logging
pytest tests/foundation/test_enhanced_retrieval.py::test_semantic_query_expansion -v --log-cli-level=DEBUG
```

### 🔍 Understanding Log Output

#### Live Console Logs
```
10:22:05 [    INFO] root: 🧪 TEST START: test_semantic_query_expansion
10:22:05 [    INFO] bot_config: [CONFIG] Successfully loaded OPENAI_API_KEY
10:22:06 [    INFO] httpx: HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
10:22:06 [    INFO] retriever: [ENHANCED_RETRIEVER] Generated 3 query alternatives
10:22:06 [    INFO] root: ✅ PASSED: tests/foundation/test_enhanced_retrieval.py::test_semantic_query_expansion
```

#### Log File Format
```
2025-07-08 10:22:05 [    INFO] root: 🧪 TEST START: test_semantic_query_expansion (conftest.py:58)
2025-07-08 10:22:05 [    INFO] bot_config: [CONFIG] Successfully loaded OPENAI_API_KEY (bot_config.py:45)
2025-07-08 10:22:06 [    INFO] httpx: HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK" (httpx.py:789)
```

### 🔧 Test Categories

#### **Fast Tests** (`./run_tests.sh fast`)
- No external dependencies (OpenAI, ChromaDB)
- Unit tests for core logic
- Quick execution (< 30 seconds)
- Ideal for development iteration

#### **Integration Tests** (`./run_tests.sh integration`)
- Requires ChromaDB connection
- Tests real document retrieval
- Longer execution time
- Full system validation

#### **Foundation Tests** (`./run_tests.sh foundation`)
- All Foundation Improvement tests (FI-01 to FI-04)
- Comprehensive coverage
- Both unit and integration tests

#### **Verbose Tests** (`./run_tests.sh verbose`)
- Maximum detail output
- Full traceback on failures
- Debug-level information
- Useful for troubleshooting

#### **Debug Tests** (`./run_tests.sh debug`)
- Interactive debugging
- Breakpoint support
- Step-through capability
- Development debugging

### 📈 Test Results Interpretation

#### Success Indicators
- ✅ **PASSED**: Test completed successfully
- 🧪 **TEST START**: Test initialization
- 🔧 **Initializing**: Service setup
- 📊 **Generated**: Successful data generation

#### Warning Indicators
- ⏭️ **SKIPPED**: Test skipped (usually due to missing dependencies)
- ⚠️ **WARNING**: Non-critical issues

#### Error Indicators
- ❌ **FAILED**: Test failed
- 💥 **ERROR**: Critical system error
- 🚨 **EXCEPTION**: Unexpected exception

### 🔍 Debugging Tips

#### 1. **Check Log Files**
```bash
# View latest log file
tail -f tests/logs/test_run_$(date +%Y%m%d)_*.log

# Search for specific errors
grep -n "ERROR\|FAILED" tests/logs/test_run_*.log
```

#### 2. **Enable Debug Mode**
```bash
# Run with Python debugger
./run_tests.sh debug

# Run single test with debug
pytest tests/foundation/test_enhanced_retrieval.py::test_semantic_query_expansion --pdb
```

#### 3. **Analyze HTTP Requests**
```bash
# Filter HTTP requests from logs
grep "httpx" tests/logs/test_run_*.log | head -20
```

#### 4. **Track Test Performance**
```bash
# View test timing
grep "TEST START\|TEST END" tests/logs/test_run_*.log
```

### 📋 Test Coverage

The enhanced retrieval system has comprehensive test coverage:

- **FI-04.1**: Semantic Query Expansion ✅
- **FI-04.2**: Multi-Vector Search Coverage ✅
- **FI-04.3**: Adaptive Similarity Thresholds ✅
- **FI-04.4**: Query Classification Accuracy ✅
- **FI-04.5**: Entity and Concept Extraction ✅
- **FI-04.6**: Enhanced vs Original Retrieval ✅
- **FI-04.7**: Learning System Integration ✅
- **Additional**: Contextual Embeddings ✅
- **Additional**: Hierarchical Search ✅
- **Additional**: Caching Functionality ✅

### 🎯 Best Practices

1. **Use Appropriate Test Categories**: Start with `fast` tests, then move to `integration`
2. **Check Log Files**: Always review detailed logs for debugging
3. **Monitor Performance**: Track test execution times and API calls
4. **Incremental Testing**: Run specific tests during development
5. **CI/CD Integration**: Use categorized tests for different pipeline stages

### 🔄 Continuous Improvement

The logging system captures:
- API call patterns and timing
- Test execution performance
- Error patterns and troubleshooting data
- System behavior under different conditions

This data helps optimize both the tests and the underlying system performance.

---

## 🚀 Quick Start

1. **Run all tests with logging**:
   ```bash
   ./run_tests.sh
   ```

2. **Check results**:
   ```bash
   ls -la tests/logs/
   ```

3. **View detailed logs**:
   ```bash
   tail -f tests/logs/test_run_$(date +%Y%m%d)_*.log
   ```

The comprehensive logging setup ensures you have complete visibility into test execution, making debugging and system validation much more effective. 