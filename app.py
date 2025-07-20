from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
from bot_config import get_openai_api_key
from prompt_template import get_prompt_template
from markdown_processor import process_markdown_to_clean_text, process_markdown_to_html, process_streaming_token
from retriever import RetrieverService
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from ratelimit import limits, sleep_and_retry
from jwt_handler import extract_jwt_claims
import asyncio
import json
import logging
import os
import uuid

load_dotenv()

# FastAPI Knowledge Bot - v2.5 Speed/Accuracy Optimized - Clean Deploy for Render

# FI-02: Semantic Topic Change Detection
async def detect_topic_change_semantic(current_question: str, chat_history: InMemoryChatMessageHistory, embedding_function) -> bool:
    """
    FI-02: Content-agnostic semantic similarity-based topic change detection.
    Uses text-embedding-3-large for topic comparison with 0.7 threshold.
    """
    if not chat_history.messages:
        return False
    
    # Get the last user message
    user_messages = [msg for msg in chat_history.messages if isinstance(msg, HumanMessage)]
    if not user_messages:
        return False
    
    last_user_message = user_messages[-1].content
    
    try:
        # Generate embeddings for current and previous questions
        current_embedding = await asyncio.to_thread(
            embedding_function.embed_query, 
            current_question
        )
        previous_embedding = await asyncio.to_thread(
            embedding_function.embed_query, 
            last_user_message
        )
        
        # Calculate cosine similarity
        import numpy as np
        current_embedding = np.array(current_embedding)
        previous_embedding = np.array(previous_embedding)
        
        similarity = np.dot(current_embedding, previous_embedding) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(previous_embedding)
        )
        
        # Return True if topics are different (low similarity)
        topic_changed = similarity < 0.7
        if topic_changed:
            logger.info(f"[FI-02] Topic change detected: similarity={similarity:.3f} < 0.7")
        
        return topic_changed
        
    except Exception as e:
        logger.error(f"[FI-02] Error in topic change detection: {e}")
        return False

app = FastAPI()

# Mount static files for widget.js
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_debug.log")
    ]
)
logger = logging.getLogger(__name__)

session_histories = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
        logger.info(f"[MEMORY] Created new history for session {session_id}")
    return session_histories[session_id]

async def format_chat_history_smart(
    chat_history: InMemoryChatMessageHistory, 
    current_question: str, 
    embedding_function, 
    complexity: str = 'complex'
) -> str:
    """
    Smart chat history formatting with semantic topic analysis.
    Provides enhanced context management based on topic continuity.
    """
    if not chat_history.messages:
        return ""
    
    # Check for topic change to optimize context
    topic_changed = await detect_topic_change_semantic(current_question, chat_history, embedding_function)
    
    if topic_changed:
        # Reduced context for topic changes to avoid contamination  
        context_limits = {'simple': 1, 'medium': 2, 'complex': 2}  # Q&A pairs * 2
        truncate_limits = {'simple': 100, 'medium': 150, 'complex': 200}
        logger.info("[SMART-HISTORY] Topic change detected - using reduced context")
    else:
        # Full context for topic continuity
        context_limits = {'simple': 2, 'medium': 4, 'complex': 6}  # Q&A pairs * 2  
        truncate_limits = {'simple': 150, 'medium': 250, 'complex': 400}
        logger.info("[SMART-HISTORY] Topic continuity - using full context")
    
    context_limit = context_limits.get(complexity, 4)
    truncate_limit = truncate_limits.get(complexity, 300)
    
    formatted = []
    for message in chat_history.messages[-context_limit:]:
        if isinstance(message, HumanMessage):
            formatted.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            # Smart truncation preserves key information  
            content = message.content
            if len(content) > truncate_limit:
                # Try to truncate at sentence boundaries for better context
                sentences = content.split('. ')
                if len(sentences) > 1:
                    truncated = '. '.join(sentences[:2]) + '.'
                    if len(truncated) <= truncate_limit:
                        content = truncated
                    else:
                        content = content[:truncate_limit] + "..."
                else:
                    content = content[:truncate_limit] + "..."
            formatted.append(f"Assistant: {content}")
    
    if formatted:
        context_type = "reduced" if topic_changed else "full"
        return f"Previous conversation ({context_type} context):\n" + "\n".join(formatted) + "\n\n"
    return ""

def format_chat_history(chat_history: InMemoryChatMessageHistory, complexity: str = 'simple') -> str:
    """Format chat history for inclusion in prompt based on query complexity"""
    if not chat_history.messages:
        return ""
    
    # Optimize context based on complexity: simple=1 pair, medium=2 pairs, complex=3 pairs  
    context_limits = {'simple': 2, 'medium': 4, 'complex': 6}  # Q&A pairs * 2
    context_limit = context_limits.get(complexity, 2)
    
    # Optimize truncation based on complexity
    truncate_limits = {'simple': 150, 'medium': 250, 'complex': 400}
    truncate_limit = truncate_limits.get(complexity, 200)
    
    formatted = []
    for message in chat_history.messages[-context_limit:]:
        if isinstance(message, HumanMessage):
            formatted.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            # Truncate AI responses based on complexity needs
            content = message.content[:truncate_limit] + "..." if len(message.content) > truncate_limit else message.content
            formatted.append(f"Assistant: {content}")
    
    if formatted:
        return "Previous conversation:\n" + "\n".join(formatted) + "\n\n"
    return ""

# Smart Complex Mode handles all queries with intelligent routing - no complexity detection needed

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class AskRequest(BaseModel):
    question: str
    k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    session_id: Optional[str] = None

class EventStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()
        self._loop = None
        self.accumulated_text = ""  # Track full response for conversation history
        self.processed_text = ""    # Clean version for history
        # FI-07: Smart Streaming Enhancement
        self.token_buffer = ""      # Buffer for word boundary detection
        self.chunk_id = 0          # Incremental chunk ID
        self.stream_started = False

    async def astream(self):
        # FI-07: Send start marker
        if not self.stream_started:
            start_chunk = {
                "id": 0,
                "type": "start",
                "content": "",
                "content_type": "start",
                "final": False
            }
            yield f"data: {json.dumps(start_chunk)}\n\n"
            self.stream_started = True
            
        while True:
            try:
                item = await self.queue.get()
                if item is None:
                    # FI-07: Send end marker
                    end_chunk = {
                        "id": self.chunk_id + 1,
                        "type": "end", 
                        "content": "",
                        "content_type": "end",
                        "final": True
                    }
                    yield f"data: {json.dumps(end_chunk)}\n\n"
                    break
                elif isinstance(item, dict):
                    # FI-07: Structured chunk
                    yield f"data: {json.dumps(item)}\n\n"
                else:
                    # Legacy token support
                    yield f"data: {item}\n\n"
            except Exception as e:
                logger.error(f"[STREAM] Error streaming token: {e}")
                error_chunk = {
                    "id": self.chunk_id + 1,
                    "type": "error",
                    "content": f"Stream error: {str(e)}",
                    "content_type": "error", 
                    "final": True
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                break

    def _put_nowait_safe(self, item):
        try:
            self.queue.put_nowait(item)
        except Exception as e:
            logger.error(f"[STREAM] Error putting item in queue: {e}")

    def _should_flush_buffer(self, new_token: str) -> bool:
        """FI-07: Determine if token buffer should be flushed at word boundaries"""
        combined = self.token_buffer + new_token
        
        # Flush on sentence boundaries
        if new_token in '.!?':
            return True
            
        # Flush on word boundaries (space after word)
        if new_token == ' ' and len(self.token_buffer.strip()) > 0:
            return True
            
        # Flush on line breaks (important for markdown)
        if '\n' in new_token:
            return True
            
        # Flush on markdown patterns
        if combined.strip().endswith('**') or combined.strip().endswith('###'):
            return True
            
        # Prevent hanging (max 50 chars)
        if len(self.token_buffer) > 50:
            return True
            
        return False

    def _classify_content_type(self, content: str) -> str:
        """FI-07: Classify content type for better client processing"""
        content_strip = content.strip()
        
        if content_strip.startswith('###') or content_strip.startswith('##') or content_strip.startswith('#'):
            return "header"
        elif content_strip.startswith('- ') or content_strip.startswith('* '):
            return "list_item" 
        elif '[source:' in content_strip:
            return "source"
        else:
            return "text"

    def on_llm_new_token(self, token: str, **kwargs):
        self.accumulated_text += token
        
        # FI-07: Smart word boundary buffering
        if self._should_flush_buffer(token):
            # Flush current buffer plus new token
            content = self.token_buffer + token
            if content.strip():  # Only send non-empty content
                self.chunk_id += 1
                chunk = {
                    "id": self.chunk_id,
                    "type": "content",
                    "content": content,
                    "content_type": self._classify_content_type(content),
                    "final": False
                }
                logger.debug(f"[FI-07] Sending chunk {self.chunk_id}: {content[:50]}...")
                self._put_nowait_safe(chunk)
            self.token_buffer = ""
        else:
            # Add to buffer
            self.token_buffer += token

    def on_llm_end(self, response: LLMResult, **kwargs):
        # FI-07: Flush any remaining buffer
        if self.token_buffer.strip():
            self.chunk_id += 1
            final_chunk = {
                "id": self.chunk_id,
                "type": "content", 
                "content": self.token_buffer,
                "content_type": self._classify_content_type(self.token_buffer),
                "final": False
            }
            self._put_nowait_safe(final_chunk)
        
        # Log the raw LLM output for debugging
        logger.info(f"[RAW LLM OUTPUT] Raw response from LLM:\n{repr(self.accumulated_text)}")
        logger.info(f"[RAW LLM OUTPUT] Raw response formatted:\n{self.accumulated_text}")
        
        # Create clean text for conversation history (remove markdown artifacts)
        self.processed_text = process_markdown_to_clean_text(self.accumulated_text)
        
        logger.info(f"[FI-07] Smart streaming completed. Chunks sent: {self.chunk_id}")
        logger.info(f"[STREAM] Completed streaming. Clean text for history: {len(self.processed_text)} chars")
        self._put_nowait_safe(None)

    def on_llm_error(self, error: Exception, **kwargs):
        error_chunk = {
            "id": self.chunk_id + 1,
            "type": "error",
            "content": f"LLM error: {str(error)}",
            "content_type": "error",
            "final": True
        }
        self._put_nowait_safe(error_chunk)

@sleep_and_retry
@limits(calls=100, period=60)
async def ask_question(
    question: str,
    company_id: int,
    bot_id: int,
    session_id: str,
    streaming: bool = False,
    stream_handler: Optional[EventStreamHandler] = None,
    k: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    use_multi_query: bool = False,
    verbose: bool = False
):
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Use qa mode for optimal performance
        mode = "qa"
        if verbose:
            logger.info(f"[BOT] Using mode: {mode}")

        # Smart Complex Mode for ALL queries - maximum performance and coverage
        logger.info("[BOT] Smart Complex Mode enabled for ALL queries")
        
        # Smart Complex Mode: Optimized for 5-6 second responses with maximum quality
        comparative_indicators = {'compare', 'versus', 'vs', 'difference between', 'analyze', 'contrast'}
        is_comparative = any(indicator in question.lower() for indicator in comparative_indicators)
        
        if is_comparative:
            logger.info("[BOT] Smart Complex → MultiQuery (comparative analysis detected)")
            k = 6  # Optimized: reduced from 8 for speed while maintaining coverage
            use_multi_query = True
        else:
            logger.info("[BOT] Smart Complex → Fast Comprehensive (comprehensive coverage)")
            k = 8  # Optimized: reduced from 12 for 5-6s responses while maintaining quality
            use_multi_query = False

        retriever_service = RetrieverService()
        
        # Use enhanced retriever with query-adaptive processing and all Foundation Improvements
        retriever = await retriever_service.build_enhanced_retriever(
            company_id=company_id,
            bot_id=bot_id,
            query=question,  # Pass query for adaptive processing
            k=k,
            similarity_threshold=similarity_threshold,
            use_multi_query=use_multi_query,
            use_enhanced_search=True  # Enable enhanced search by default
        )
        
        # Enhanced Retrieval Debug System - Test retrieval quality and coverage
        if verbose:
            try:
                logger.info(f"[RETRIEVAL DEBUG] Testing enhanced retrieval for query: '{question}'")
                test_docs = retriever.get_relevant_documents(question)
                logger.info(f"[RETRIEVAL DEBUG] Retrieved {len(test_docs)} documents")
                
                # Show top 3 documents with metadata
                for i, doc in enumerate(test_docs[:3]):
                    content_preview = doc.page_content[:150].replace('\n', ' ')
                    logger.info(f"[RETRIEVAL DEBUG] Doc {i+1}: {content_preview}...")
                    
                    metadata = doc.metadata
                    file_info = {
                        'file_name': metadata.get('file_name', 'Unknown'),
                        'chunk_index': metadata.get('chunk_index', 'N/A'),
                        'source_type': metadata.get('source_type', 'Unknown')
                    }
                    logger.info(f"[RETRIEVAL DEBUG] Metadata: {file_info}")
                
                # Log document source diversity
                sources = [doc.metadata.get('file_name', 'Unknown') for doc in test_docs]
                unique_sources = list(set(sources))
                logger.info(f"[RETRIEVAL DEBUG] Source diversity: {len(unique_sources)} unique sources from {len(test_docs)} documents")
                
            except Exception as e:
                logger.error(f"[RETRIEVAL DEBUG] Error testing retrieval: {e}")

        # Use comprehensive prompt template for Smart Complex mode
        prompt = get_prompt_template(mode)

        # Enhanced document prompt with person context for better attribution
        def create_enhanced_document_prompt():
            def format_document(doc):
                page_content = doc.page_content
                file_name = doc.metadata.get('file_name', 'Unknown')
                chunk_index = doc.metadata.get('chunk_index', '')
                
                # Extract person context from filename for better attribution
                person_context = ""
                filename_lower = file_name.lower()
                
                # Content-agnostic person detection (avoiding hardcoded names)
                if any(keyword in filename_lower for keyword in ['resume', 'cv']):
                    person_context = "FROM PERSONAL RESUME: "
                elif len(filename_lower.split()) > 1:
                    # Check if filename contains typical personal document patterns  
                    personal_indicators = ['personal', 'profile', 'bio', 'portfolio']
                    if any(indicator in filename_lower for indicator in personal_indicators):
                        person_context = "FROM PERSONAL DOCUMENT: "
                
                source_ref = f"{file_name}#{chunk_index}" if chunk_index else file_name
                return f"{person_context}{page_content}\n[source: {source_ref}]"
            
            # Custom document prompt that processes each document with person context
            class EnhancedDocumentPrompt:
                def format(self, **kwargs):
                    docs = kwargs.get('summaries', [])
                    if not docs:
                        return ""
                    
                    formatted_docs = []
                    for doc in docs:
                        if hasattr(doc, 'page_content'):
                            formatted_docs.append(format_document(doc))
                        else:
                            # Fallback for string documents
                            formatted_docs.append(str(doc))
                    
                    return "\n\n".join(formatted_docs)
            
            return EnhancedDocumentPrompt()
        
        # Use enhanced document prompt for better source attribution
        document_prompt = PromptTemplate.from_template(
            "{page_content}\n[source: {file_name}#{chunk_index}]"
        )

        # Use optimal model for Smart Complex mode
        model = 'gpt-4o-mini'  # Best quality for comprehensive analysis
        if verbose:
            logger.info(f"[BOT] Using {model} for Smart Complex query optimization")
            
        llm = ChatOpenAI(
            model=model,
            temperature=0,
            streaming=streaming,
            callbacks=[stream_handler] if streaming else None,
            openai_api_key=get_openai_api_key()
        )

        # Get conversation history for this session - Smart Complex mode with enhanced context
        chat_history = get_session_history(session_id)
        
        # Smart conversation context with semantic topic change detection
        embedding_function = retriever_service.embedding_function
        conversation_context = await format_chat_history_smart(
            chat_history, question, embedding_function, 'complex'
        )
        
        # Apply maximum context limits for Smart Complex mode
        max_context = 1200  # Full context for Smart Complex mode
        if len(conversation_context) > max_context:
            conversation_context = conversation_context[:max_context] + "...\n\n"
        
        enhanced_prompt_template = conversation_context + prompt.template
        enhanced_prompt = PromptTemplate(
            template=enhanced_prompt_template,
            input_variables=prompt.input_variables
        )
        
        # Create a simple, reliable chain with conversational context
        from langchain.chains import RetrievalQA
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=verbose,
            chain_type_kwargs={
                "prompt": enhanced_prompt,
                "document_prompt": document_prompt
            }
        )

        if streaming:
            async def run_chain():
                try:
                    result = await asyncio.to_thread(
                        chain.invoke,
                        {"query": question}
                    )
                    
                    # Save conversation to history using post-processed clean text
                    if stream_handler and stream_handler.processed_text:
                        chat_history.add_user_message(question)
                        chat_history.add_ai_message(stream_handler.processed_text)
                    
                    sources = result.get("source_documents", [])
                    if sources:
                        # Add widget-compatible source markers
                        for doc in sources:
                            filename = doc.metadata.get('file_name', 'Unknown')
                            chunk_idx = doc.metadata.get('chunk_index', '')
                            source_ref = f"{filename}#{chunk_idx}" if chunk_idx else filename
                            await stream_handler.queue.put(f"[source: {source_ref}]")
                        scores = [doc.metadata.get('score', 'N/A') for doc in sources]
                        logger.info(f"[RETRIEVAL] Sources: {[doc.metadata.get('file_name') for doc in sources]}, Scores: {scores}")
                except Exception as e:
                    logger.error(f"[STREAM] Error in chain: {e}")
                    if stream_handler:
                        await stream_handler.queue.put(f"[ERROR] {str(e)}")
                finally:
                    if stream_handler:
                        await stream_handler.queue.put(None)
            return run_chain()
        else:
            result = await asyncio.to_thread(
                chain.invoke,
                {"query": question}
            )
            
            # Save conversation to history with post-processed clean text
            answer = result.get("result", "")
            if answer:
                processed_answer = process_markdown_to_clean_text(answer)
                chat_history.add_user_message(question)
                chat_history.add_ai_message(processed_answer)
                logger.info(f"[MARKDOWN] Non-streaming post-processed: {len(answer)} → {len(processed_answer)} chars")
            
            sources = result.get("source_documents", [])
            scores = [doc.metadata.get('score', 'N/A') for doc in sources]
            logger.info(f"[RETRIEVAL] Sources: {[doc.metadata.get('file_name') for doc in sources]}, Scores: {scores}")
            return result

    except Exception as e:
        logger.error(f"[BOT] Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Main API endpoint with JWT authentication and real LLM streaming
@app.post("/ask")
async def ask_api(request: AskRequest, jwt_claims: dict = Depends(extract_jwt_claims)):
    session_id = request.session_id or str(uuid.uuid4())
    company_id = jwt_claims['company_id']
    bot_id = jwt_claims['bot_id']
    
    logger.info(f"[API] Received question: {request.question}, company_id: {company_id}, bot_id: {bot_id}, session_id: {session_id}")
    
    async def generate_stream():
        try:
            
            # Set up real streaming handler
            stream_handler = EventStreamHandler()
            
            # Start LLM processing in background (this will feed the stream_handler)
            async def run_llm_chain():
                try:
                    result = await ask_question(
                        question=request.question,
                        company_id=company_id,
                        bot_id=bot_id,
                        session_id=session_id,
                        streaming=True,
                        stream_handler=stream_handler,
                        k=request.k,
                        similarity_threshold=request.similarity_threshold,
                        use_multi_query=False,  # Auto-detection handles this internally
                        verbose=True
                    )
                    # Get the actual coroutine result
                    if hasattr(result, '__await__'):
                        await result
                except Exception as e:
                    logger.error(f"[STREAM] Chain error: {e}")
                    await stream_handler.queue.put(f"[ERROR] {str(e)}")
                    await stream_handler.queue.put(None)
            
            # Start the LLM chain
            llm_task = asyncio.create_task(run_llm_chain())
            
            # Stream tokens as they come from the LLM in real-time
            async for chunk in stream_handler.astream():
                yield chunk
                
            # Ensure the task completes
            await llm_task
                    
        except Exception as e:
            logger.error(f"[API] Streaming error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"X-Session-ID": session_id}
    )

# Serve widget.js
@app.get("/widget.js")
async def serve_widget():
    widget_path = os.path.join("static", "widget.js")
    if not os.path.exists(widget_path):
        raise HTTPException(status_code=404, detail="widget.js not found")
    return FileResponse(widget_path, media_type="application/javascript")

# Smart Complex stats endpoint 
@app.get("/complexity-stats")
async def get_complexity_stats():
    """Get Smart Complex mode statistics for performance monitoring"""
    return {
        "message": "Smart Complex Mode is hardcoded for ALL queries with enhanced coverage",
        "mode": "Smart Complex Enhanced",
        "routing": {
            "comparative_queries": "MultiQuery (k=8) for enhanced analysis",
            "standard_queries": "Fast Complex (k=12) for comprehensive coverage"
        },
        "triggers": ["compare", "versus", "vs", "difference between", "analyze", "contrast"],
        "performance": "3-5 second streaming start with comprehensive coverage",
        "coverage": "12+ sources for standard queries, 8+ sources for comparative queries"
    }
