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
import logging
import os
import uuid
import numpy as np

load_dotenv()

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

# Smart Complex Mode - All queries use intelligent routing

# Smart Complex Mode is now hardcoded for all queries

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
        logger.info(f"[MEMORY] Created new history for session {session_id}")
    return session_histories[session_id]

async def detect_topic_change_semantic(current_question: str, chat_history: InMemoryChatMessageHistory, embedding_function) -> bool:
    """
    Content-agnostic topic change detection using semantic similarity.
    Detects when current question is semantically different from recent conversation.
    """
    if not chat_history.messages:
        return False
    
    # Get the last user question for comparison
    last_user_message = None
    for message in reversed(chat_history.messages):
        if isinstance(message, HumanMessage):
            last_user_message = message.content
            break
    
    if not last_user_message:
        return False
    
    try:
        # Use semantic similarity to detect topic changes
        current_embedding = await asyncio.to_thread(
            embedding_function.embed_query, current_question
        )
        previous_embedding = await asyncio.to_thread(
            embedding_function.embed_query, last_user_message
        )
        
        # Calculate cosine similarity
        similarity = np.dot(current_embedding, previous_embedding) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(previous_embedding)
        )
        
        # Threshold for topic change detection - tunable based on testing
        # Lower similarity indicates different topics
        topic_change_threshold = 0.7  # Conservative threshold
        topic_changed = similarity < topic_change_threshold
        
        if topic_changed:
            logger.info(f"[TOPIC] Semantic topic change detected (similarity: {similarity:.3f} < {topic_change_threshold})")
            logger.info(f"[TOPIC] Previous: '{last_user_message[:100]}...'")
            logger.info(f"[TOPIC] Current: '{current_question[:100]}...'")
        else:
            logger.info(f"[TOPIC] Same topic detected (similarity: {similarity:.3f})")
        
        return topic_changed
        
    except Exception as e:
        logger.error(f"[TOPIC] Error in semantic topic detection: {e}")
        # Fallback: assume no topic change if detection fails
        return False

async def format_chat_history_smart(chat_history: InMemoryChatMessageHistory, current_question: str, embedding_function, complexity: str = 'simple') -> str:
    """
    Format chat history with smart semantic topic change detection.
    Reduces or eliminates conversation context when topic changes are detected.
    """
    if not chat_history.messages:
        return ""
    
    # Check for topic change using semantic similarity
    topic_changed = await detect_topic_change_semantic(current_question, chat_history, embedding_function)
    
    if topic_changed:
        logger.info("[CONTEXT] Topic change detected - using minimal conversation context")
        # For topic changes, only include the most recent exchange (if any) with reduced context
        context_limits = {'simple': 0, 'medium': 2, 'complex': 2}  # Much more conservative
        truncate_limits = {'simple': 50, 'medium': 100, 'complex': 150}  # Shorter context
    else:
        logger.info("[CONTEXT] Same topic - using full conversation context")
        # Original behavior for same topic
        context_limits = {'simple': 2, 'medium': 4, 'complex': 6}
        truncate_limits = {'simple': 150, 'medium': 250, 'complex': 400}
    
    context_limit = context_limits.get(complexity, 2)
    truncate_limit = truncate_limits.get(complexity, 200)
    
    if context_limit == 0:
        # No conversation context for topic changes in simple mode
        return ""
    
    formatted = []
    for message in chat_history.messages[-context_limit:]:
        if isinstance(message, HumanMessage):
            formatted.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            # Truncate AI responses based on complexity needs
            content = message.content[:truncate_limit] + "..." if len(message.content) > truncate_limit else message.content
            formatted.append(f"Assistant: {content}")
    
    if formatted:
        context_prefix = "Previous conversation:\n" if not topic_changed else "Recent context:\n"
        return context_prefix + "\n".join(formatted) + "\n\n"
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

class SmartEventStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()
        self._loop = None
        self.accumulated_text = ""  # Track full response for conversation history
        self.processed_text = ""    # Clean version for history
        self.token_buffer = ""      # Buffer for intelligent boundary streaming
        self.chunk_id = 0          # Unique chunk identifier
        self.streaming_started = False

    async def astream(self):
        while True:
            try:
                chunk_data = await self.queue.get()
                if chunk_data is None:
                    # Send final end-of-stream marker
                    yield f"data: {self._format_json_chunk('', 'end', final=True)}\n\n"
                    break
                yield f"data: {chunk_data}\n\n"
            except Exception as e:
                logger.error(f"[STREAM] Error streaming chunk: {e}")
                yield f"data: {self._format_json_chunk('', 'error', error=str(e))}\n\n"
                break

    def _format_json_chunk(self, content: str, chunk_type: str = 'content', 
                          final: bool = False, error: str = None) -> str:
        """Format chunk as proper JSON structure"""
        import json
        
        self.chunk_id += 1
        chunk_data = {
            "id": self.chunk_id,
            "type": chunk_type,
            "content": content,
            "final": final
        }
        
        if error:
            chunk_data["error"] = error
            
        if chunk_type == 'content' and content:
            # Analyze content type for better client handling
            if content.strip().startswith(('#', '##', '###')):
                chunk_data["content_type"] = "header"
            elif content.strip().startswith(('-', '*', '+')):
                chunk_data["content_type"] = "list_item"
            elif '[source:' in content:
                chunk_data["content_type"] = "source"
            else:
                chunk_data["content_type"] = "text"
        
        return json.dumps(chunk_data, ensure_ascii=False)

    def _put_nowait_safe(self, item):
        try:
            self.queue.put_nowait(item)
        except Exception as e:
            logger.error(f"[STREAM] Error putting item in queue: {e}")

    def _should_flush_buffer(self, new_token: str) -> bool:
        """Determine if we should flush the current buffer"""
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
            
        # Flush if buffer gets too long (prevent hanging)
        if len(self.token_buffer) > 50:
            return True
            
        return False

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts generating"""
        if not self.streaming_started:
            self.streaming_started = True
            # Send start marker
            start_chunk = self._format_json_chunk('', 'start')
            self._put_nowait_safe(start_chunk)

    def on_llm_new_token(self, token: str, **kwargs):
        self.accumulated_text += token
        self.token_buffer += token
        
        # Check if we should flush the buffer
        if self._should_flush_buffer(token):
            if self.token_buffer.strip():  # Only send non-empty content
                content_chunk = self._format_json_chunk(self.token_buffer, 'content')
                self._put_nowait_safe(content_chunk)
                logger.debug(f"[STREAM] Sent chunk: '{self.token_buffer.strip()[:50]}'...")
            self.token_buffer = ""
        
        # For very short tokens, send immediately to maintain responsiveness
        elif len(token) == 1 and token in '.,!?;:':
            if self.token_buffer.strip():
                content_chunk = self._format_json_chunk(self.token_buffer, 'content')
                self._put_nowait_safe(content_chunk)
                logger.debug(f"[STREAM] Sent punctuation chunk: '{self.token_buffer.strip()}'")
            self.token_buffer = ""

    def on_llm_end(self, response: LLMResult, **kwargs):
        # Flush any remaining buffer
        if self.token_buffer.strip():
            content_chunk = self._format_json_chunk(self.token_buffer, 'content')
            self._put_nowait_safe(content_chunk)
            logger.debug(f"[STREAM] Sent final buffer: '{self.token_buffer.strip()[:50]}'...")
        
        # Log the raw LLM output for debugging
        logger.info(f"[RAW LLM OUTPUT] Raw response from LLM:\n{repr(self.accumulated_text)}")
        logger.info(f"[RAW LLM OUTPUT] Raw response formatted:\n{self.accumulated_text}")
        
        # Create clean text for conversation history (remove markdown artifacts)
        self.processed_text = process_markdown_to_clean_text(self.accumulated_text)
        
        logger.info(f"[STREAM] Completed streaming. Clean text for history: {len(self.processed_text)} chars")
        logger.info(f"[STREAM] Total chunks sent: {self.chunk_id}")
        
        # Send end marker
        self._put_nowait_safe(None)

    def on_llm_error(self, error: Exception, **kwargs):
        error_chunk = self._format_json_chunk('', 'error', error=str(error))
        self._put_nowait_safe(error_chunk)

@sleep_and_retry
@limits(calls=100, period=60)
async def ask_question(
    question: str,
    company_id: int,
    bot_id: int,
    session_id: str,
    streaming: bool = False,
    stream_handler: Optional[SmartEventStreamHandler] = None,
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
        
        # Smart Complex Mode: Use MultiQuery for comparative/analytical queries, Fast for others
        comparative_indicators = {'compare', 'versus', 'vs', 'difference between', 'analyze', 'contrast'}
        is_comparative = any(indicator in question.lower() for indicator in comparative_indicators)
        
        if is_comparative:
            logger.info("[BOT] Smart Complex → MultiQuery (comparative analysis detected)")
            k = 8  # Increased from 6 for better coverage
            use_multi_query = True
        else:
            logger.info("[BOT] Smart Complex → Fast Comprehensive (comprehensive coverage)")
            k = 12  # Increased from 8 for better coverage
            use_multi_query = False

        retriever_service = RetrieverService()
        
        # Use enhanced retriever with content-agnostic improvements
        retriever = await retriever_service.build_enhanced_retriever(
            company_id=company_id,
            bot_id=bot_id,
            query=question,  # Pass the query for adaptive processing
            k=k,
            similarity_threshold=similarity_threshold,
            use_multi_query=use_multi_query,
            use_enhanced_search=True  # Enable enhanced search by default
        )

        # Use comprehensive prompt template for Smart Complex mode
        prompt = get_prompt_template(mode)

        # Enhanced document prompt with person context - creates clear ownership indicators
        document_prompt_template = """{% if 'vishal' in file_name.lower() %}FROM VISHAL'S DOCUMENT: {% elif 'bremer' in file_name.lower() or 'marty' in file_name.lower() %}FROM MARTY'S DOCUMENT: {% elif 'resume' in file_name.lower() or 'cv' in file_name.lower() %}FROM PERSONAL RESUME: {% endif %}{page_content}
[source: {file_name}#{chunk_index}]"""
        
        # Create enhanced document prompt with conditional formatting
        def create_enhanced_document_prompt():
            def format_document(page_content, file_name, chunk_index):
                # Extract person name from filename for better context
                person_context = ""
                filename_lower = file_name.lower()
                if 'vishal' in filename_lower:
                    person_context = "FROM VISHAL'S DOCUMENT: "
                elif 'bremer' in filename_lower or 'marty' in filename_lower:
                    person_context = "FROM MARTY'S DOCUMENT: "
                elif 'resume' in filename_lower or 'cv' in filename_lower:
                    person_context = "FROM PERSONAL RESUME: "
                
                source_ref = f"{file_name}#{chunk_index}" if chunk_index else file_name
                return f"{person_context}{page_content}\n[source: {source_ref}]"
            
            return PromptTemplate(
                input_variables=["page_content", "file_name", "chunk_index"],
                template="{person_context}{page_content}\n[source: {file_name}#{chunk_index}]"
            )
        
        # Use standard document prompt but with enhanced context in the main prompt
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

        # Get conversation history for this session - Smart Complex mode with semantic topic change detection
        chat_history = get_session_history(session_id)
        conversation_context = await format_chat_history_smart(chat_history, question, retriever_service.embedding_function, 'complex')
        
        # Enhanced prompt with smart conversation context - comprehensive mode
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
        
        # Debug logging for retrieval testing
        logger.info(f"[RETRIEVAL DEBUG] Testing retrieval for query: '{question}'")
        try:
            test_docs = retriever.get_relevant_documents(question)
            logger.info(f"[RETRIEVAL DEBUG] Retrieved {len(test_docs)} documents:")
            for i, doc in enumerate(test_docs[:3]):  # Show first 3 docs
                logger.info(f"[RETRIEVAL DEBUG] Doc {i}: {doc.page_content[:150]}...")
                logger.info(f"[RETRIEVAL DEBUG] Metadata: {doc.metadata.get('file_name', 'Unknown')}")
        except Exception as e:
            logger.error(f"[RETRIEVAL DEBUG] Error testing retrieval: {e}")

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
            stream_handler = SmartEventStreamHandler()
            
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
        "message": "Smart Complex Mode is hardcoded for ALL queries with improved retrieval",
        "mode": "Smart Complex Enhanced",
        "routing": {
            "comparative_queries": "MultiQuery (k=8) for enhanced analysis",
            "standard_queries": "Fast Complex (k=12) for comprehensive coverage"
        },
        "triggers": ["compare", "versus", "vs", "difference between", "analyze", "contrast"],
        "performance": "3-5 second streaming start with comprehensive coverage",
        "coverage": "12+ sources for standard queries, 8+ sources for comparative queries",
        "improvements": {
            "similarity_threshold": "Lowered to 0.05 for broader document matching",
            "bm25_weight": "Increased to 0.4 for enhanced keyword matching",
            "semantic_enhancement": "Improved semantic variation and synonym handling",
            "topic_change_detection": "Semantic similarity-based topic change detection (content-agnostic)"
        }
    }

@app.get("/test-smart-stream")
async def test_smart_stream():
    """Test endpoint for the new smart streaming format"""
    
    async def generate_test_stream():
        # Simulate the new smart streaming format
        import json
        import asyncio
        
        # Start chunk
        start_chunk = {
            "id": 1,
            "type": "start",
            "content": "",
            "final": False
        }
        yield f"data: {json.dumps(start_chunk)}\n\n"
        await asyncio.sleep(0.5)
        
        # Content chunks with different types
        chunks = [
            ("### Smart Streaming Demo", "header"),
            ("\n\nThis is a test of the new ", "text"),
            ("**smart streaming**", "text"),
            (" implementation with ", "text"),
            ("word-boundary buffering", "text"),
            (" and proper JSON structure.", "text"),
            ("\n\n- **Feature 1**: Token boundary detection", "list_item"),
            ("\n- **Feature 2**: JSON chunk format", "list_item"), 
            ("\n- **Feature 3**: Metadata support", "list_item"),
            ("\n\n### Additional Information", "header"),
            ("\n\nThis demonstrates improved streaming quality ", "text"),
            ("and better user experience.", "text"),
            ("\n\n[source: test.pdf#1]", "source")
        ]
        
        chunk_id = 1
        for content, content_type in chunks:
            chunk_id += 1
            chunk_data = {
                "id": chunk_id,
                "type": "content",
                "content": content,
                "content_type": content_type,
                "final": False
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.2)  # Simulate streaming delay
        
        # End chunk
        end_chunk = {
            "id": chunk_id + 1,
            "type": "end",
            "content": "",
            "final": True
        }
        yield f"data: {json.dumps(end_chunk)}\n\n"
    
    return StreamingResponse(
        generate_test_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
