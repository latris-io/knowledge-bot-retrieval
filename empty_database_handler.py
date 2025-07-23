#!/usr/bin/env python3
"""
Empty Database Handler for Knowledge Bot Retrieval Service
Provides graceful handling when ChromaDB is empty or has no documents
"""

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmptyDatabaseHandler:
    """Handles empty database scenarios gracefully"""
    
    @staticmethod
    def create_empty_database_response(question: str, company_id: int, bot_id: int) -> str:
        """Generate a helpful response when database is empty"""
        
        response = f"""I don't have access to any documents in my knowledge base for this query.

### 📋 **Database Status**
- Company ID: {company_id}
- Bot ID: {bot_id}
- Documents found: 0

### 🔧 **Next Steps**
1. **Upload Documents**: Please ensure relevant documents have been uploaded to the knowledge base
2. **Check Indexing**: Verify that documents have been properly processed and indexed
3. **Verify Configuration**: Confirm that the correct company and bot IDs are being used

### 💡 **What You Can Do**
- Upload documents through the ingestion service
- Check the document upload logs for any processing errors
- Verify your authentication credentials are correct

Once documents are uploaded and indexed, I'll be able to help answer questions about their content."""

        return response

    @staticmethod
    def create_empty_retriever(company_id: int, bot_id: int) -> BaseRetriever:
        """Create a retriever that returns empty results safely"""
        
        class EmptyRetriever(BaseRetriever):
            def __init__(self, company_id: int, bot_id: int, **kwargs):
                super().__init__(**kwargs)
                self.company_id = company_id
                self.bot_id = bot_id
                
            def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
                logger.warning(f"[EMPTY-DB] No documents available for company_id={self.company_id}, bot_id={self.bot_id}")
                return []
                
            async def aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
                return self._get_relevant_documents(query, run_manager=run_manager)
        
        return EmptyRetriever(company_id=company_id, bot_id=bot_id)

    @staticmethod
    def wrap_chain_for_empty_database(chain, question: str, company_id: int, bot_id: int):
        """Wrap a LangChain chain to handle empty database gracefully"""
        
        class EmptyDatabaseChain:
            def __init__(self, original_chain, question: str, company_id: int, bot_id: int):
                self.original_chain = original_chain
                self.question = question
                self.company_id = company_id
                self.bot_id = bot_id
            
            def invoke(self, inputs: dict) -> dict:
                try:
                    result = self.original_chain.invoke(inputs)
                    return result
                except Exception as e:
                    if "division by zero" in str(e).lower():
                        logger.warning(f"[EMPTY-DB] Division by zero error caught - likely empty database")
                        empty_response = EmptyDatabaseHandler.create_empty_database_response(
                            self.question, self.company_id, self.bot_id
                        )
                        return {
                            "result": empty_response,
                            "source_documents": []
                        }
                    else:
                        raise e
        
        return EmptyDatabaseChain(chain, question, company_id, bot_id)

# Easy-to-use decorators for empty database handling
def handle_empty_database(func):
    """Decorator to handle empty database scenarios"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ZeroDivisionError as e:
            logger.error(f"[EMPTY-DB] Division by zero error - database likely empty: {e}")
            return []
        except Exception as e:
            if "division by zero" in str(e).lower():
                logger.error(f"[EMPTY-DB] Division by zero error in {func.__name__}: {e}")
                return []
            else:
                raise e
    return wrapper 