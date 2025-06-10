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
    Comprehensive query complexity analysis for any industry/domain.
    Returns: 'simple', 'medium', or 'complex'
    
    Examples across industries:
    
    SIMPLE (3s): Direct factual lookups
    - "What is John's email address?"
    - "When is the meeting scheduled?"
    - "Is the product in stock?"
    - "Who is the CEO of Microsoft?"
    - "What does API stand for?"
    - "How much does this cost?"
    - "Where is your office located?"
    - "Are you open on Sundays?"
    
    MEDIUM (3s): Single-topic exploration, lists
    - "What services do you offer?"
    - "How do I apply for a loan?"
    - "List all the features of this software"
    - "What are the requirements for this job?"
    - "Explain the refund policy"
    - "Show me your pricing plans"
    - "What projects has Sarah worked on?"
    - "How does this machine work?"
    
    COMPLEX (15-20s): Multi-faceted analysis, comprehensive overviews
    - "Tell me about the company's financial performance and market strategy"
    - "Compare the advantages and disadvantages of these two approaches"
    - "Provide a comprehensive analysis of industry trends and opportunities"
    - "What are the factors I should consider when choosing between options A and B?"
    - "Explain the complete workflow from design to implementation"
    - "Give me a detailed breakdown of risks, costs, and timeline"
    - "Analyze the competitive landscape and recommend a strategy"
    """
    question_lower = question.lower().strip()
    
    # COMPLEX: Multi-faceted analysis, strategic, comprehensive overviews
    complex_patterns = [
        # Comprehensive analysis requests
        'tell me about', 'give me an overview', 'provide a comprehensive', 'detailed analysis',
        'complete picture', 'full breakdown', 'thorough review', 'in-depth look',
        
        # Multiple aspects combined (using "and")
        ' and ', 'as well as', 'along with', 'including', 'plus', 'together with',
        'combined with', 'in addition to', 'not only', 'both',
        
        # Strategic/planning queries
        'strategy', 'plan for', 'approach to', 'roadmap', 'framework', 'methodology',
        'best practices', 'recommendations', 'suggestions', 'advice', 'guidance',
        
        # Comparative analysis
        'compare', 'comparison', 'versus', 'vs', 'difference between', 'better than',
        'advantages', 'disadvantages', 'pros and cons', 'strengths and weaknesses',
        
        # Trend/market analysis
        'trends', 'market analysis', 'industry outlook', 'forecast', 'predictions',
        'future of', 'evolution', 'development', 'growth', 'opportunities',
        
        # Research-style queries
        'research', 'study', 'investigate', 'examine', 'analyze', 'evaluate',
        'assess', 'review', 'survey', 'comprehensive study',
        
        # Multi-dimensional requests
        'factors', 'considerations', 'aspects', 'dimensions', 'elements',
        'components', 'variables', 'parameters', 'criteria',
        
        # Industry-specific complex analysis
        'market conditions', 'competitive landscape', 'swot analysis', 'risk assessment',
        'financial analysis', 'business case', 'feasibility study', 'impact analysis',
        'performance review', 'due diligence', 'strategic planning', 'operational efficiency',
        
        # Healthcare/Medical complex
        'treatment options', 'side effects', 'contraindications', 'drug interactions',
        'differential diagnosis', 'prognosis', 'clinical outcomes', 'patient care',
        
        # Legal complex  
        'legal implications', 'regulatory compliance', 'contract terms', 'liability',
        'intellectual property', 'litigation risk', 'legal framework', 'statutory requirements',
        
        # Technical/Engineering complex
        'system architecture', 'technical specifications', 'integration challenges',
        'scalability issues', 'performance optimization', 'security considerations',
        
        # Educational complex
        'learning outcomes', 'curriculum design', 'assessment strategies', 'pedagogical approaches',
        'student performance', 'educational effectiveness', 'learning methodologies'
    ]
    
    # MEDIUM: Single-topic exploration, lists, moderate analysis
    medium_patterns = [
        # List requests
        'list', 'what are', 'show me', 'give me', 'provide', 'enumerate',
        'identify', 'name', 'specify', 'outline', 'detail',
        
        # Single-topic exploration
        'explain', 'describe', 'how does', 'how to', 'why', 'what makes',
        'process', 'procedure', 'steps', 'method', 'way to', 'approach',
        
        # Professional/business single topics
        'experience', 'background', 'qualifications', 'skills', 'expertise',
        'projects', 'achievements', 'accomplishments', 'work history',
        'education', 'training', 'certifications', 'credentials',
        
        # Industry-specific single topics
        'services', 'products', 'offerings', 'solutions', 'capabilities',
        'features', 'benefits', 'applications', 'uses', 'functions',
        
        # Technical explanations
        'specifications', 'requirements', 'standards', 'guidelines',
        'protocols', 'policies', 'regulations', 'compliance',
        
        # Financial/business metrics
        'costs', 'pricing', 'budget', 'revenue', 'profit', 'expenses',
        'performance', 'metrics', 'results', 'outcomes', 'statistics',
        
        # Process/workflow queries
        'workflow', 'timeline', 'schedule', 'phases', 'stages',
        'milestones', 'deliverables', 'objectives', 'goals',
        
        # Industry-specific medium complexity
        'inventory', 'catalog', 'portfolio', 'menu', 'roster', 'directory',
        'curriculum', 'syllabus', 'agenda', 'itinerary', 'program',
        
        # Healthcare/Medical medium
        'symptoms', 'treatments', 'medications', 'procedures', 'tests',
        'diagnoses', 'specialists', 'departments', 'facilities',
        
        # Legal medium
        'laws', 'regulations', 'statutes', 'ordinances', 'codes',
        'procedures', 'forms', 'documents', 'requirements', 'filings',
        
        # Technical/IT medium
        'features', 'functions', 'tools', 'utilities', 'modules',
        'components', 'interfaces', 'APIs', 'integrations', 'plugins',
        
        # Finance/Business medium
        'rates', 'fees', 'charges', 'packages', 'plans', 'options',
        'terms', 'conditions', 'policies', 'procedures', 'guidelines',
        
        # Real Estate medium
        'properties', 'listings', 'amenities', 'neighborhoods', 'schools',
        'transportation', 'utilities', 'taxes', 'HOA', 'restrictions',
        
        # Manufacturing/Supply Chain medium
        'materials', 'suppliers', 'vendors', 'distributors', 'logistics',
        'equipment', 'machinery', 'tools', 'resources', 'capacity',
        
        # Education medium
        'courses', 'programs', 'degrees', 'majors', 'requirements',
        'faculty', 'staff', 'departments', 'resources', 'facilities'
    ]
    
    # SIMPLE: Direct factual lookups, single pieces of information
    simple_patterns = [
        # Basic identification
        'who is', 'what is', 'when is', 'where is', 'which is',
        'who are', 'what are', 'when are', 'where are', 'which are',
        
        # Contact information
        'phone', 'email', 'address', 'contact', 'location', 'website',
        'fax', 'telephone', 'mobile', 'cell', 'number', 'reach',
        
        # Operating information
        'hours', 'schedule', 'open', 'closed', 'available', 'timing',
        'when open', 'operating hours', 'business hours', 'availability',
        
        # Yes/no questions
        'is ', 'are ', 'does ', 'do ', 'can ', 'will ', 'has ', 'have ',
        'was ', 'were ', 'did ', 'would ', 'could ', 'should ',
        
        # Single attribute queries
        'title', 'position', 'role', 'job', 'department', 'division',
        'age', 'date', 'time', 'year', 'month', 'day', 'size', 'type',
        
        # Status checks
        'status', 'state', 'condition', 'available', 'in stock',
        'active', 'inactive', 'current', 'latest', 'recent',
        
        # Single measurements/values
        'price', 'cost', 'fee', 'rate', 'amount', 'quantity', 'count',
        'weight', 'height', 'length', 'width', 'volume', 'capacity',
        
        # Simple definitions
        'definition', 'meaning', 'what does', 'stands for', 'refers to',
        'means', 'defines', 'represents', 'indicates', 'signifies',
        
        # Location/direction queries
        'directions', 'how to get', 'find', 'located', 'situated',
        'based', 'headquarters', 'office', 'branch', 'facility',
        
        # Industry-specific simple factual queries
        'license number', 'registration', 'certification', 'accreditation',
        'model number', 'part number', 'serial number', 'version',
        'account number', 'policy number', 'reference number', 'ID',
        
        # Financial simple
        'balance', 'payment', 'due date', 'interest rate', 'APR',
        'minimum payment', 'credit limit', 'available credit',
        
        # Healthcare simple
        'doctor', 'physician', 'nurse', 'appointment', 'prescription',
        'insurance', 'copay', 'deductible', 'coverage',
        
        # Legal simple
        'lawyer', 'attorney', 'case number', 'court date', 'filing fee',
        'legal aid', 'consultation', 'retainer',
        
        # Technical simple
        'username', 'password', 'login', 'account', 'support',
        'version number', 'build', 'release date', 'compatibility',
        
        # Real Estate simple
        'square footage', 'bedrooms', 'bathrooms', 'lot size',
        'property tax', 'MLS number', 'listing agent', 'asking price',
        
        # Employment simple
        'salary', 'wages', 'benefits', 'vacation days', 'sick leave',
        'supervisor', 'manager', 'HR', 'employee ID',
        
        # Education simple
        'tuition', 'enrollment', 'registration', 'GPA', 'credits',
        'semester', 'quarter', 'graduation date', 'transcript',
        
        # Travel/Hospitality simple
        'reservation', 'booking', 'check-in', 'check-out', 'cancellation',
        'confirmation number', 'flight number', 'gate', 'terminal',
        
        # Retail simple
        'SKU', 'barcode', 'warranty', 'return policy', 'exchange',
        'store hours', 'pickup', 'delivery', 'shipping'
    ]
    
    # Multi-word phrase detection for complex queries
    complex_multi_word = [
        'tell me everything about', 'give me a complete overview of',
        'what can you tell me about', 'i want to know about',
        'provide information on', 'i need details on',
        'comprehensive analysis of', 'detailed breakdown of',
        'full explanation of', 'complete guide to'
    ]
    
    # Check for complex multi-word phrases first
    for phrase in complex_multi_word:
        if phrase in question_lower:
            return 'complex'
    
    # Count complexity indicators
    complex_count = sum(1 for pattern in complex_patterns if pattern in question_lower)
    medium_count = sum(1 for pattern in medium_patterns if pattern in question_lower)
    simple_count = sum(1 for pattern in simple_patterns if pattern in question_lower)
    
    # Multiple complex indicators = definitely complex
    if complex_count >= 2:
        return 'complex'
    
    # Question length-based complexity (longer questions tend to be more complex)
    word_count = len(question_lower.split())
    if word_count > 15:
        if complex_count >= 1:
            return 'complex'
        elif medium_count >= 1:
            return 'medium'
    elif word_count > 8:
        if complex_count >= 1:
            return 'complex'
    
    # Pattern-based classification
    if complex_count > 0:
        return 'complex'
    elif medium_count > 0:
        return 'medium'
    elif simple_count > 0:
        return 'simple'
    
    # Default classification based on question structure
    question_starters_complex = ['explain how', 'describe how', 'what are the factors',
                                'how can i', 'what should i', 'help me understand']
    question_starters_medium = ['how do', 'what kind', 'which type', 'how many',
                               'what sort', 'how much', 'how long']
    question_starters_simple = ['what', 'who', 'when', 'where', 'which', 'is', 'are',
                               'does', 'do', 'can', 'will', 'has', 'have']
    
    for starter in question_starters_complex:
        if question_lower.startswith(starter):
            return 'complex'
    
    for starter in question_starters_medium:
        if question_lower.startswith(starter):
            return 'medium'
    
    for starter in question_starters_simple:
        if question_lower.startswith(starter):
            return 'simple'
    
    # Final fallback: default to simple for speed
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
