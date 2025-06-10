#!/usr/bin/env python3
"""
Debug Brentwood office hours discrepancy
Analyze what's in the knowledge base vs what's being retrieved
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from retriever import RetrieverService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

async def debug_brentwood_office():
    """Debug the Brentwood office hours issue"""
    
    company_id = 3
    bot_id = 1
    query = "when is the brentwood office open"
    
    print("🔍 DEBUGGING BRENTWOOD OFFICE HOURS DISCREPANCY")
    print("=" * 60)
    
    try:
        retriever_service = RetrieverService()
        vectorstore = retriever_service.get_chroma_vectorstore("global")
        
        # Step 1: Find all documents for this company/bot
        base_filter = {
            "$and": [
                {"company_id": {"$eq": company_id}},
                {"bot_id": {"$eq": bot_id}}
            ]
        }
        
        all_docs = vectorstore.get(include=["documents", "metadatas"], where=base_filter)
        print(f"\n📊 TOTAL DOCUMENTS: {len(all_docs['documents'])}")
        
        # Step 2: Look for Lucas Office.pdf specifically
        lucas_docs = []
        for i, (doc_content, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
            file_name = metadata.get('file_name', '').lower()
            if 'lucas' in file_name and 'office' in file_name:
                lucas_docs.append((i, doc_content, metadata))
                
        print(f"\n📄 LUCAS OFFICE DOCUMENTS FOUND: {len(lucas_docs)}")
        
        if lucas_docs:
            print("\n🔍 LUCAS OFFICE DOCUMENT CONTENTS:")
            print("-" * 50)
            for idx, (doc_idx, content, metadata) in enumerate(lucas_docs):
                print(f"\nDocument {idx + 1}:")
                print(f"   File: {metadata.get('file_name', 'Unknown')}")
                print(f"   Chunk: {metadata.get('chunk_index', 'Unknown')}")
                print(f"   Size: {len(content)} characters")
                print("   Content Preview:")
                print(f"   {content[:500]}...")
                
                # Look for Brentwood specifically
                if 'brentwood' in content.lower():
                    print(f"\n   ✅ CONTAINS BRENTWOOD:")
                    lines = content.split('\n')
                    for line_num, line in enumerate(lines):
                        if 'brentwood' in line.lower():
                            # Show context around Brentwood mention
                            start = max(0, line_num - 2)
                            end = min(len(lines), line_num + 10)
                            context_lines = lines[start:end]
                            print("   Context:")
                            for i, ctx_line in enumerate(context_lines):
                                marker = ">>>" if i == (line_num - start) else "   "
                                print(f"   {marker} {ctx_line}")
                            break
        else:
            print("❌ No Lucas Office documents found!")
            
        # Step 3: Test the actual retrieval query
        print(f"\n🎯 TESTING QUERY: '{query}'")
        print("-" * 50)
        
        retriever = retriever_service.build_retriever(
            company_id=company_id,
            bot_id=bot_id,
            k=20,
            similarity_threshold=0.1
        )
        
        retrieved_docs = retriever.invoke(query)
        print(f"📊 RETRIEVED: {len(retrieved_docs)} documents")
        
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                metadata = doc.metadata
                content = doc.page_content
                
                print(f"\n📄 Retrieved Doc {i+1}:")
                print(f"   File: {metadata.get('file_name', 'Unknown')}")
                print(f"   Chunk: {metadata.get('chunk_index', 'Unknown')}")
                print(f"   Content: {content[:300]}...")
                
                # Look for office hours in retrieved content
                if any(time in content.lower() for time in ['7:00', '7:30', '6:30', '4:00', '4:30', '3:30']):
                    print(f"   ✅ CONTAINS OFFICE HOURS")
                    # Extract lines with times
                    lines = content.split('\n')
                    for line in lines:
                        if any(time in line for time in ['7:00', '7:30', '6:30', '4:00', '4:30', '3:30']):
                            print(f"      📅 {line.strip()}")
                            
        # Step 4: Search for all documents containing Brentwood
        print(f"\n🔍 ALL DOCUMENTS CONTAINING 'BRENTWOOD':")
        print("-" * 50)
        
        brentwood_docs = []
        for i, (doc_content, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
            if 'brentwood' in doc_content.lower():
                brentwood_docs.append((i, doc_content, metadata))
                
        print(f"Found {len(brentwood_docs)} documents with 'Brentwood'")
        
        for idx, (doc_idx, content, metadata) in enumerate(brentwood_docs):
            print(f"\nBrentwood Doc {idx + 1}:")
            print(f"   File: {metadata.get('file_name', 'Unknown')}")
            print(f"   Chunk: {metadata.get('chunk_index', 'Unknown')}")
            
            # Extract Brentwood office hours specifically
            lines = content.split('\n')
            in_brentwood_section = False
            for line in lines:
                line_lower = line.lower()
                if 'brentwood' in line_lower:
                    in_brentwood_section = True
                    print(f"   📍 {line.strip()}")
                elif in_brentwood_section and any(day in line_lower for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
                    print(f"   📅 {line.strip()}")
                elif in_brentwood_section and line.strip() == "":
                    continue
                elif in_brentwood_section and any(time in line for time in ['7:00', '7:30', '6:30', '4:00', '4:30', '3:30']):
                    print(f"   📅 {line.strip()}")
                elif in_brentwood_section and line.strip() and not any(char.isdigit() for char in line):
                    # Probably end of section
                    break
                    
        print(f"\n🎯 ANALYSIS COMPLETE")
        print("Compare the 'LUCAS OFFICE DOCUMENT CONTENTS' with 'RETRIEVED' docs")
        print("Look for discrepancies in the office hours data.")
        
    except Exception as e:
        logger.error(f"❌ Error during debugging: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(debug_brentwood_office()) 