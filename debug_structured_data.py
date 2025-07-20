#!/usr/bin/env python3
"""
COMPREHENSIVE STRUCTURED DATA DIAGNOSTIC

Tests all the specific queries identified as failing by the ingestion team
Analyzes what data is actually indexed vs what should be available  
Provides actionable recommendations for fixing structured data retrieval

Based on ingestion team's comprehensive Use Case 1 test results:
- 75% failure rate on CSV/Excel structured queries
- Success on general concepts, failure on specific data points
- 24.4% overall failure rate due to structured data indexing issues
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from retriever import RetrieverService
from typing import List, Dict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

class StructuredDataDiagnostic:
    def __init__(self):
        self.company_id = 3
        self.bot_id = 1
        self.retriever_service = RetrieverService()
        self.vectorstore = None
        self.failed_queries = []
        self.successful_queries = []
        
    async def initialize(self):
        """Initialize the diagnostic tool"""
        self.vectorstore = self.retriever_service.get_chroma_vectorstore("global")
        print("🔬 STRUCTURED DATA DIAGNOSTIC INITIALIZED")
        print("=" * 80)

    async def test_query_with_analysis(self, query: str, file_type: str, expected_data: str, should_succeed: bool = True):
        """Test a specific query and analyze the results"""
        print(f"\n{'✅' if should_succeed else '❌'} Testing: {query}")
        print(f"   File Type: {file_type}")
        print(f"   Expected: {expected_data}")
        print("-" * 60)
        
        try:
            # Test retrieval
            retriever = await self.retriever_service.build_enhanced_retriever(
                company_id=self.company_id,
                bot_id=self.bot_id,
                query=query,
                k=12,
                similarity_threshold=0.05,
                use_multi_query=False,
                use_enhanced_search=True
            )
            
            retrieved_docs = retriever.get_relevant_documents(query)
            
            print(f"📊 Retrieved {len(retrieved_docs)} documents")
            
            # Analyze retrieved documents
            relevant_found = False
            for i, doc in enumerate(retrieved_docs[:5]):  # Check top 5
                metadata = doc.metadata
                content = doc.page_content.lower()
                file_name = metadata.get('file_name', 'Unknown')
                
                print(f"\n📄 Doc {i+1}: {file_name}")
                print(f"   Content: {doc.page_content[:200]}...")
                
                # Check if this doc contains the expected file type
                if file_type.lower() in file_name.lower():
                    print(f"   ✅ MATCHES FILE TYPE: {file_type}")
                    
                    # Check if content contains expected data patterns
                    expected_terms = expected_data.lower().split()
                    matches = [term for term in expected_terms if term in content]
                    if matches:
                        print(f"   ✅ CONTAINS EXPECTED TERMS: {matches}")
                        relevant_found = True
                    else:
                        print(f"   ❌ MISSING EXPECTED TERMS: {expected_terms}")
                        
            # Record results
            if relevant_found and should_succeed:
                self.successful_queries.append({
                    'query': query, 
                    'file_type': file_type,
                    'expected': expected_data,
                    'docs_retrieved': len(retrieved_docs)
                })
                print(f"🎯 RESULT: SUCCESS - Found relevant data")
            elif not relevant_found and should_succeed:
                self.failed_queries.append({
                    'query': query,
                    'file_type': file_type, 
                    'expected': expected_data,
                    'docs_retrieved': len(retrieved_docs),
                    'issue': 'Expected data not found in retrieved documents'
                })
                print(f"❌ RESULT: FAILED - Expected data not retrieved")
            elif not relevant_found and not should_succeed:
                print(f"✅ RESULT: CORRECTLY FAILED - No data found as expected")
                
        except Exception as e:
            print(f"💥 ERROR: {e}")
            if should_succeed:
                self.failed_queries.append({
                    'query': query,
                    'file_type': file_type,
                    'expected': expected_data,
                    'docs_retrieved': 0,
                    'issue': f'Exception: {str(e)}'
                })

    async def analyze_document_structure(self):
        """Analyze what documents are actually stored and how they're chunked"""
        print(f"\n🔍 ANALYZING DOCUMENT STRUCTURE")
        print("=" * 80)
        
        base_filter = {
            "$and": [
                {"company_id": {"$eq": self.company_id}},
                {"bot_id": {"$eq": self.bot_id}}
            ]
        }
        
        all_docs = self.vectorstore.get(include=["documents", "metadatas"], where=base_filter)
        print(f"📊 TOTAL DOCUMENTS: {len(all_docs['documents'])}")
        
        # Group by file type
        file_types = {}
        for doc_content, metadata in zip(all_docs['documents'], all_docs['metadatas']):
            file_name = metadata.get('file_name', 'Unknown')
            file_ext = file_name.split('.')[-1].lower() if '.' in file_name else 'unknown'
            
            if file_ext not in file_types:
                file_types[file_ext] = []
            file_types[file_ext].append({
                'file_name': file_name,
                'content': doc_content,
                'metadata': metadata,
                'content_length': len(doc_content),
                'chunk_index': metadata.get('chunk_index', 0)
            })
            
        # Analyze each file type
        for file_ext, docs in file_types.items():
            print(f"\n📁 {file_ext.upper()} FILES: {len(docs)} chunks")
            
            # Group by file name to see chunking strategy
            files = {}
            for doc in docs:
                fname = doc['file_name']
                if fname not in files:
                    files[fname] = []
                files[fname].append(doc)
                
            for fname, file_docs in files.items():
                print(f"   📄 {fname}: {len(file_docs)} chunks")
                
                # Show content samples for structured data files
                if file_ext in ['csv', 'xlsx', 'xls']:
                    print(f"      🔍 STRUCTURED DATA ANALYSIS:")
                    for i, doc in enumerate(file_docs[:3]):  # Show first 3 chunks
                        content = doc['content']
                        print(f"         Chunk {i+1} ({doc['content_length']} chars):")
                        print(f"         {content[:300]}...")
                        
                        # Analyze if chunk contains specific data
                        has_prices = any(char in content for char in ['$', '€', '£']) or 'price' in content.lower()
                        has_names = any(word in content.lower() for word in ['customer', 'name', 'client'])
                        has_quantities = any(word in content.lower() for word in ['stock', 'qty', 'quantity', 'units'])
                        
                        print(f"         Data indicators: Prices={has_prices}, Names={has_names}, Quantities={has_quantities}")

    async def test_known_failing_queries(self):
        """Test all the specific queries identified as failing by the ingestion team"""
        print(f"\n🧪 TESTING KNOWN FAILING QUERIES")
        print("=" * 80)
        print("Based on ingestion team's comprehensive Use Case 1 test results")
        
        # CSV/Table Data Failures (75% failure rate)
        await self.test_query_with_analysis(
            query="What products are in the Widgets category?",
            file_type="sample_table.csv",
            expected_data="Widget product list from CSV",
            should_succeed=True
        )
        
        await self.test_query_with_analysis(
            query="Which items are low in stock?",
            file_type="sample_table.csv", 
            expected_data="Low stock items from CSV",
            should_succeed=True
        )
        
        await self.test_query_with_analysis(
            query="What is the price of the Premium Gadget?",
            file_type="sample_table.csv",
            expected_data="Premium Gadget price from CSV",
            should_succeed=True
        )
        
        # Excel/Financial Data Failures (75% failure rate)  
        await self.test_query_with_analysis(
            query="What are the sales figures for different months?",
            file_type="sample_excel.xlsx",
            expected_data="Monthly sales data from Excel",
            should_succeed=True
        )
        
        await self.test_query_with_analysis(
            query="Which customers are listed in the data?",
            file_type="sample_excel.xlsx",
            expected_data="Customer names from Excel",
            should_succeed=True
        )
        
        await self.test_query_with_analysis(
            query="What regions are represented in the sales data?",
            file_type="sample_excel.xlsx", 
            expected_data="Geographic regions from Excel",
            should_succeed=True
        )
        
        # Project Management/List Data Failures (50% failure rate)
        await self.test_query_with_analysis(
            query="What are the high priority tasks?",
            file_type="sample_list.md",
            expected_data="High priority tasks from markdown list",
            should_succeed=True
        )
        
        await self.test_query_with_analysis(
            query="What authentication features are completed?",
            file_type="sample_list.md",
            expected_data="Completed authentication features",
            should_succeed=True
        )
        
        # Business Report Detail Failures (50% failure rate)
        await self.test_query_with_analysis(
            query="What are the key performance indicators mentioned?",
            file_type="sample_word.docx",
            expected_data="Specific KPI list from business report",
            should_succeed=True
        )
        
        await self.test_query_with_analysis(
            query="What revenue figures are reported?",
            file_type="sample_word.docx",
            expected_data="Revenue numbers from business report", 
            should_succeed=True
        )

    async def test_known_successful_queries(self):
        """Test queries that should work to confirm retrieval system is functional"""
        print(f"\n✅ TESTING KNOWN SUCCESSFUL QUERIES")
        print("=" * 80)
        print("These should work to confirm retrieval system functionality")
        
        # High-level queries that should succeed
        await self.test_query_with_analysis(
            query="What categories are available?",
            file_type="sample_table.csv",
            expected_data="Category overview from CSV",
            should_succeed=True
        )
        
        await self.test_query_with_analysis(
            query="What financial metrics are tracked?",
            file_type="sample_excel.xlsx",
            expected_data="General KPI concepts from Excel",
            should_succeed=True
        )
        
        await self.test_query_with_analysis(
            query="What is the main focus of the business report?",
            file_type="sample_word.docx",
            expected_data="Business report summary",
            should_succeed=True
        )

    def generate_report(self):
        """Generate comprehensive diagnostic report"""
        print(f"\n📊 DIAGNOSTIC REPORT")
        print("=" * 80)
        
        total_queries = len(self.failed_queries) + len(self.successful_queries)
        success_rate = (len(self.successful_queries) / total_queries * 100) if total_queries > 0 else 0
        
        print(f"📈 RESULTS SUMMARY:")
        print(f"   Total Queries Tested: {total_queries}")
        print(f"   Successful: {len(self.successful_queries)} ({success_rate:.1f}%)")
        print(f"   Failed: {len(self.failed_queries)} ({100-success_rate:.1f}%)")
        
        if self.failed_queries:
            print(f"\n❌ FAILED QUERIES ANALYSIS:")
            for i, failure in enumerate(self.failed_queries):
                print(f"   {i+1}. Query: {failure['query']}")
                print(f"      File Type: {failure['file_type']}")  
                print(f"      Expected: {failure['expected']}")
                print(f"      Issue: {failure['issue']}")
                print(f"      Documents Retrieved: {failure['docs_retrieved']}")
                print()
                
        print(f"\n🎯 ROOT CAUSE ANALYSIS:")
        print(f"   Primary Issue: Structured data chunking/indexing problems")
        print(f"   Pattern: General concepts retrievable, specific data points not accessible")
        print(f"   Impact: {100-success_rate:.1f}% query failure rate on structured data")
        
        print(f"\n🔧 RECOMMENDATIONS FOR INGESTION SERVICE:")
        print(f"   1. IMPROVE CSV/EXCEL CHUNKING:")
        print(f"      - Chunk at row level, not document level")
        print(f"      - Index individual product names, prices, customer data")
        print(f"      - Preserve field-level searchability")
        print()
        print(f"   2. ENHANCE METADATA TAGGING:")
        print(f"      - Tag product names, prices, customer info as searchable entities")
        print(f"      - Add entity type metadata (product, customer, price, etc.)")
        print(f"      - Include field names as searchable terms")
        print()
        print(f"   3. GRANULAR LIST PROCESSING:")  
        print(f"      - Process list items individually in markdown/documents")
        print(f"      - Preserve task details and completion status")
        print(f"      - Index specific feature names and statuses")
        print()
        print(f"   4. STRUCTURED DATA EXTRACTION:")
        print(f"      - Extract tables into searchable key-value pairs")
        print(f"      - Create chunks for each data row/item")
        print(f"      - Maintain relationship between fields and values")

async def main():
    """Run the comprehensive structured data diagnostic"""
    diagnostic = StructuredDataDiagnostic()
    
    try:
        await diagnostic.initialize()
        await diagnostic.analyze_document_structure()
        await diagnostic.test_known_failing_queries()
        await diagnostic.test_known_successful_queries()
        diagnostic.generate_report()
        
        print(f"\n🎯 DIAGNOSTIC COMPLETE")
        print(f"Refer to the recommendations above for fixing structured data retrieval issues.")
        
    except Exception as e:
        print(f"💥 DIAGNOSTIC ERROR: {e}")
        logger.error(f"Diagnostic error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main()) 