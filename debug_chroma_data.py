#!/usr/bin/env python3
"""
Debug ChromaDB Data
Check what documents are actually stored in the database
"""

import asyncio
import logging
from dotenv import load_dotenv
from retriever import RetrieverService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_chroma_data():
    """Debug what data is actually in ChromaDB"""
    
    # Test with the same IDs as the diagnostic
    company_id = 3
    bot_id = 1
    
    print("🔍 DEBUGGING CHROMADB DATA")
    print("=" * 60)
    print(f"Company ID: {company_id}")
    print(f"Bot ID: {bot_id}")
    
    try:
        retriever_service = RetrieverService()
        vectorstore = retriever_service.get_chroma_vectorstore("global")
        
        # Step 1: Check total documents in database
        try:
            all_docs = vectorstore.get(include=["documents", "metadatas"])
            print(f"\n📊 TOTAL DOCUMENTS IN DATABASE: {len(all_docs['documents'])}")
            
            if all_docs['documents']:
                # Show a few sample documents
                print(f"\n📄 SAMPLE DOCUMENTS:")
                for i, (doc, meta) in enumerate(zip(all_docs['documents'][:3], all_docs['metadatas'][:3])):
                    print(f"   {i+1}. Company: {meta.get('company_id', 'N/A')}, Bot: {meta.get('bot_id', 'N/A')}")
                    print(f"      File: {meta.get('file_name', 'Unknown')}")
                    print(f"      Content: {doc[:100]}...")
                    print()
            
        except Exception as e:
            print(f"❌ Error getting all documents: {e}")
            
        # Step 2: Check documents for specific company/bot
        base_filter = {
            "$and": [
                {"company_id": {"$eq": company_id}},
                {"bot_id": {"$eq": bot_id}}
            ]
        }
        
        try:
            filtered_docs = vectorstore.get(include=["documents", "metadatas"], where=base_filter)
            print(f"📋 DOCUMENTS FOR COMPANY {company_id}, BOT {bot_id}: {len(filtered_docs['documents'])}")
            
            if filtered_docs['documents']:
                print(f"\n✅ FOUND DOCUMENTS:")
                for i, (doc, meta) in enumerate(zip(filtered_docs['documents'][:5], filtered_docs['metadatas'][:5])):
                    print(f"   {i+1}. File: {meta.get('file_name', 'Unknown')}")
                    print(f"      Chunk: {meta.get('chunk_index', 'N/A')}")
                    print(f"      Content: {doc[:150]}...")
                    print()
            else:
                print(f"❌ NO DOCUMENTS FOUND FOR COMPANY {company_id}, BOT {bot_id}")
                print(f"\n🔍 Checking what company/bot IDs exist:")
                
                # Find what company/bot combinations exist
                if all_docs['metadatas']:
                    combinations = set()
                    for meta in all_docs['metadatas']:
                        combo = (meta.get('company_id'), meta.get('bot_id'))
                        combinations.add(combo)
                    
                    print(f"Available combinations:")
                    for combo in sorted(combinations):
                        print(f"   Company: {combo[0]}, Bot: {combo[1]}")
                        
        except Exception as e:
            print(f"❌ Error getting filtered documents: {e}")
            
        # Step 3: Test retrieval
        print(f"\n🧪 TESTING RETRIEVAL:")
        try:
            retriever = retriever_service.build_retriever(
                company_id=company_id,
                bot_id=bot_id,
                k=5
            )
            
            test_docs = retriever.get_relevant_documents("What industries are represented?")
            print(f"   Retrieved: {len(test_docs)} documents")
            
            if test_docs:
                print(f"   ✅ Retrieval working!")
                for i, doc in enumerate(test_docs[:2]):
                    print(f"      {i+1}. {doc.page_content[:100]}...")
            else:
                print(f"   ❌ No documents retrieved")
                
        except Exception as e:
            print(f"   💥 Retrieval error: {e}")
            
    except Exception as e:
        print(f"💥 MAJOR ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_chroma_data()) 