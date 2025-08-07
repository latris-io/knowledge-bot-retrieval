#!/usr/bin/env python3
"""
Test Sample Data Upload
Upload sample documents to ChromaDB to test streaming functionality
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from retriever import RetrieverService
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def upload_sample_data():
    """Upload sample documents to test streaming"""
    
    # Use the same IDs as diagnostic
    company_id = 3
    bot_id = 1
    
    print("📤 UPLOADING SAMPLE DATA TO CHROMADB")
    print("=" * 60)
    print(f"Company ID: {company_id}")
    print(f"Bot ID: {bot_id}")
    
    try:
        retriever_service = RetrieverService()
        vectorstore = retriever_service.get_chroma_vectorstore("global")
        
        # Sample documents to test streaming
        sample_docs = [
            Document(
                page_content="TechCorp is a leading technology company specializing in artificial intelligence and machine learning solutions. Founded in 2020, the company operates in the Technology industry with offices in San Francisco and New York.",
                metadata={
                    "file_name": "sample_company_data.txt",
                    "chunk_index": 0,
                    "company_id": company_id,
                    "bot_id": bot_id,
                    "source_type": "txt"
                }
            ),
            Document(
                page_content="HealthPlus operates in the Healthcare industry, providing medical software solutions for hospitals and clinics. The company offers cloud-based patient management systems and electronic health records.",
                metadata={
                    "file_name": "sample_company_data.txt", 
                    "chunk_index": 1,
                    "company_id": company_id,
                    "bot_id": bot_id,
                    "source_type": "txt"
                }
            ),
            Document(
                page_content="FinanceFirst is a Financial Services company that provides investment management and banking solutions. They serve both individual and corporate clients with a focus on sustainable investing.",
                metadata={
                    "file_name": "sample_company_data.txt",
                    "chunk_index": 2, 
                    "company_id": company_id,
                    "bot_id": bot_id,
                    "source_type": "txt"
                }
            ),
            Document(
                page_content="GreenEnergy Solutions operates in the Renewable Energy industry, developing solar and wind power projects across the United States. The company was established in 2018.",
                metadata={
                    "file_name": "sample_company_data.txt",
                    "chunk_index": 3,
                    "company_id": company_id, 
                    "bot_id": bot_id,
                    "source_type": "txt"
                }
            ),
            Document(
                page_content="RetailMax is a major player in the Retail industry, operating both online and physical stores. They specialize in consumer electronics and home appliances.",
                metadata={
                    "file_name": "sample_company_data.txt",
                    "chunk_index": 4,
                    "company_id": company_id,
                    "bot_id": bot_id, 
                    "source_type": "txt"
                }
            )
        ]
        
        print(f"\n📝 Uploading {len(sample_docs)} sample documents...")
        
        # Upload documents
        texts = [doc.page_content for doc in sample_docs]
        metadatas = [doc.metadata for doc in sample_docs] 
        
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
        
        print(f"✅ Successfully uploaded {len(sample_docs)} documents!")
        
        # Verify upload
        print(f"\n🔍 Verifying upload...")
        base_filter = {
            "$and": [
                {"company_id": {"$eq": company_id}},
                {"bot_id": {"$eq": bot_id}}
            ]
        }
        
        stored_docs = vectorstore.get(include=["documents", "metadatas"], where=base_filter)
        print(f"📊 Documents now in database: {len(stored_docs['documents'])}")
        
        if stored_docs['documents']:
            print(f"\n📄 Sample content:")
            for i, doc in enumerate(stored_docs['documents'][:2]):
                print(f"   {i+1}. {doc[:100]}...")
                
        print(f"\n🎉 Ready to test streaming! Try the diagnostic again.")
        
    except Exception as e:
        print(f"💥 ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(upload_sample_data()) 