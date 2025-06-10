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

def get_query_complexity(question: str) -> str:
    """
    Determine query complexity level for optimal processing strategy.
    Returns: 'simple', 'medium', or 'complex'
    """
    question_lower = question.lower().strip()
    
    # High complexity: Multiple aspects, comprehensive analysis
    complex_patterns = [
        'tell me about', 'describe', 'overview', 'summary', 'explain',
        'projects and experience', 'background and', 'history and',
        'achievements and', 'capabilities and', 'services and',
        'compare', 'difference', 'versus', 'vs', 'better', 'best'
    ]
    
    # Medium complexity: Lists, single broad topics
    medium_patterns = [
        'list', 'what are', 'show me', 'projects', 'experience', 
        'background', 'history', 'achievements', 'services', 
        'offerings', 'capabilities', 'expertise', 'how does', 'why'
    ]
    
    # Simple: Direct factual questions
    simple_patterns = [
        'who is', 'what is', 'when is', 'where is',
        'phone', 'email', 'address', 'contact', 'hours', 'schedule',
        'is', 'does', 'can', 'will', 'has'
    ]
    
    # Check for complex patterns first (most comprehensive)
    if any(pattern in question_lower for pattern in complex_patterns):
        return 'complex'
    
    # Check for medium patterns (moderate coverage)
    if any(pattern in question_lower for pattern in medium_patterns):
        return 'medium'
    
    # Check for simple patterns (direct retrieval)
    if any(pattern in question_lower for pattern in simple_patterns):
        return 'simple'
    
    # Default: simple for speed (most queries are specific)
    return 'simple'

def should_use_multi_query(question: str) -> bool:
    """Determine if MultiQueryRetriever should be used (complex queries only)"""
    return get_query_complexity(question) == 'complex'

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

        # Three-tier intelligent complexity detection
        complexity = get_query_complexity(question)
        auto_multi_query = should_use_multi_query(question)  # Only for complex
        is_simple_query = complexity == 'simple'
        
        if verbose:
            strategies = {
                'simple': "Direct (maximum speed)",
                'medium': "Enhanced Direct (balanced speed+quality)", 
                'complex': "MultiQuery (comprehensive coverage)"
            }
            logger.info(f"[BOT] Auto-detected complexity: {complexity} → {strategies[complexity]}")

        # Optimize k value based on complexity tier
        if k is None:
            k_values = {'simple': 2, 'medium': 4, 'complex': 6}  # Graduated approach
            k = k_values[complexity]
            if verbose:
                logger.info(f"[BOT] Optimized k={k} for {complexity} query")

        retriever_service = RetrieverService()
        retriever = retriever_service.build_retriever(
            company_id=company_id,
            bot_id=bot_id,
            k=k,
            similarity_threshold=similarity_threshold,
            use_multi_query=auto_multi_query
        )

        # Optimize prompt based on complexity level
        if complexity == 'simple':
            # Concise prompt for maximum speed on simple factual queries
            concise_template = """Answer the question using the context below. Be accurate and concise.

Context:
{context}

Question:
{question}

Answer:"""
            prompt = PromptTemplate(input_variables=["context", "question"], template=concise_template)
        elif complexity == 'medium':
            # Balanced prompt for medium complexity
            medium_template = """Use the context below to answer the question accurately. Provide helpful details while being concise.

Context:
{context}

Question:
{question}

Answer:"""
            prompt = PromptTemplate(input_variables=["context", "question"], template=medium_template)
        else:
            # Full detailed prompt for complex queries
            prompt = get_prompt_template(mode)

        document_prompt = PromptTemplate.from_template(
            "{page_content}\n[source: {file_name}#{chunk_index}]"
        )

        # Optimize model selection based on complexity
        model_map = {
            'simple': 'gpt-3.5-turbo',      # Fastest for factual queries
            'medium': 'gpt-3.5-turbo',      # Fast but good quality for moderate complexity  
            'complex': 'gpt-4o-mini'        # Best quality for comprehensive analysis
        }
        model = model_map[complexity]
        if verbose:
            logger.info(f"[BOT] Using {model} for {complexity} query optimization")
            
        llm = ChatOpenAI(
            model=model,
            temperature=0,
            streaming=streaming,
            callbacks=[stream_handler] if streaming else None,
            openai_api_key=get_openai_api_key()
        )

        # Get conversation history for this session - optimize based on complexity
        chat_history = get_session_history(session_id)
        if complexity == 'simple' and not chat_history.messages:
            # Skip conversation context entirely for simple queries with no history
            conversation_context = ""
        else:
            conversation_context = format_chat_history(chat_history, complexity)
        
        # Enhanced prompt with conversation context - optimized for performance
        context_limits = {'simple': 400, 'medium': 800, 'complex': 1200}
        max_context = context_limits.get(complexity, 500)
        if len(conversation_context) > max_context:
            # Truncate conversation context based on complexity to optimize processing time
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
