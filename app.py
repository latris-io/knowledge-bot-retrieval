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

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
        logger.info(f"[MEMORY] Created new history for session {session_id}")
    return session_histories[session_id]

def format_chat_history(chat_history: InMemoryChatMessageHistory, is_simple_query: bool = False) -> str:
    """Format chat history for inclusion in prompt"""
    if not chat_history.messages:
        return ""
    
    # For simple queries, use minimal context (1 Q&A pair); for complex, use more (3 Q&A pairs)
    context_limit = 2 if is_simple_query else 6  # 1 vs 3 Q&A pairs
    
    formatted = []
    for message in chat_history.messages[-context_limit:]:
        if isinstance(message, HumanMessage):
            formatted.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            # Truncate long AI responses for context efficiency
            content = message.content[:200] + "..." if len(message.content) > 200 else message.content
            formatted.append(f"Assistant: {content}")
    
    if formatted:
        return "Previous conversation:\n" + "\n".join(formatted) + "\n\n"
    return ""

def should_use_multi_query(question: str) -> bool:
    """
    Intelligently determine if a question would benefit from MultiQueryRetriever.
    Uses MultiQuery for complex/broad queries, Direct for simple/specific ones.
    """
    question_lower = question.lower().strip()
    
    # Patterns that benefit from multiple query perspectives
    broad_patterns = [
        # List/enumeration requests
        'list', 'what are', 'show me', 'tell me about', 'describe',
        # Comparison/analysis
        'compare', 'difference', 'versus', 'vs', 'better', 'best',
        # Broad exploration
        'overview', 'summary', 'explain', 'how does', 'why',
        # Multiple aspects
        'projects', 'experience', 'background', 'history', 'achievements',
        'services', 'offerings', 'capabilities', 'expertise'
    ]
    
    # Patterns that work well with direct retrieval
    specific_patterns = [
        # Direct factual questions
        'who is', 'what is', 'when is', 'where is',
        # Specific details
        'phone', 'email', 'address', 'contact', 'hours', 'schedule',
        # Simple yes/no or factual
        'is', 'does', 'can', 'will', 'has'
    ]
    
    # Check for broad patterns (use MultiQuery)
    if any(pattern in question_lower for pattern in broad_patterns):
        return True
    
    # Check for specific patterns (use Direct)
    if any(pattern in question_lower for pattern in specific_patterns):
        return False
    
    # Default: use Direct for speed (most queries are specific)
    return False

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

    async def astream(self):
        while True:
            try:
                token = await self.queue.get()
                if token is None:
                    break
                yield f"data: {token}\n\n"
            except Exception as e:
                logger.error(f"[STREAM] Error streaming token: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"
                break

    def _put_nowait_safe(self, item):
        try:
            self.queue.put_nowait(item)
        except Exception as e:
            logger.error(f"[STREAM] Error putting item in queue: {e}")

    def on_llm_new_token(self, token: str, **kwargs):
        self.accumulated_text += token
        self._put_nowait_safe(token)

    def on_llm_end(self, response: LLMResult, **kwargs):
        self._put_nowait_safe(None)

    def on_llm_error(self, error: Exception, **kwargs):
        self._put_nowait_safe(f"[ERROR] LLM failed: {str(error)}")

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

        # Intelligent auto-detection: use MultiQuery for complex/broad queries
        auto_multi_query = should_use_multi_query(question)
        is_simple_query = not auto_multi_query  # Simple query = Direct mode
        
        if verbose:
            strategy = "MultiQuery (enhanced coverage)" if auto_multi_query else "Direct (maximum speed)"
            logger.info(f"[BOT] Auto-selected retrieval strategy: {strategy}")

        # Optimize k value for performance: fewer documents for simple queries
        if k is None:
            k = 8 if auto_multi_query else 2  # Complex: 8 docs, Simple: 2 docs for maximum speed
            if verbose:
                logger.info(f"[BOT] Optimized k={k} for {'complex' if auto_multi_query else 'simple'} query")

        retriever_service = RetrieverService()
        retriever = retriever_service.build_retriever(
            company_id=company_id,
            bot_id=bot_id,
            k=k,
            similarity_threshold=similarity_threshold,
            use_multi_query=auto_multi_query
        )

        # Optimize prompt for performance: use concise prompt for simple queries
        if is_simple_query:
            # Concise prompt for maximum speed on simple factual queries
            concise_template = """Answer the question using the context below. Be accurate and concise.

Context:
{context}

Question:
{question}

Answer:"""
            prompt = PromptTemplate(input_variables=["context", "question"], template=concise_template)
        else:
            # Full detailed prompt for complex queries
            prompt = get_prompt_template(mode)

        document_prompt = PromptTemplate.from_template(
            "{page_content}\n[source: {file_name}#{chunk_index}]"
        )

        # Use faster model for simple queries, higher quality for complex ones
        model = "gpt-3.5-turbo" if is_simple_query else "gpt-4o-mini"
        if verbose and is_simple_query:
            logger.info(f"[BOT] Using {model} for maximum speed on simple query")
            
        llm = ChatOpenAI(
            model=model,
            temperature=0,
            streaming=streaming,
            callbacks=[stream_handler] if streaming else None,
            openai_api_key=get_openai_api_key()
        )

        # Get conversation history for this session - skip for simple queries if no history
        chat_history = get_session_history(session_id)
        if is_simple_query and not chat_history.messages:
            # Skip conversation context entirely for simple queries with no history
            conversation_context = ""
        else:
            conversation_context = format_chat_history(chat_history, is_simple_query)
        
        # Enhanced prompt with conversation context - optimized for performance
        if is_simple_query and len(conversation_context) > 500:
            # Truncate conversation context for simple queries to reduce processing time
            conversation_context = conversation_context[:500] + "...\n\n"
        
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
                    
                    # Save conversation to history using accumulated streaming text
                    if stream_handler and stream_handler.accumulated_text:
                        chat_history.add_user_message(question)
                        chat_history.add_ai_message(stream_handler.accumulated_text)
                    
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
            
            # Save conversation to history
            answer = result.get("result", "")
            if answer:
                chat_history.add_user_message(question)
                chat_history.add_ai_message(answer)
            
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
