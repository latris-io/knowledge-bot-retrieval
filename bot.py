# knowledge-bot-ingestion-service/bot.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import LLMResult
from ratelimit import limits, sleep_and_retry
import asyncio
import logging
import os
import uuid

load_dotenv()
app = FastAPI()

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

class AskRequest(BaseModel):
    question: str
    company_id: int
    bot_id: int
    k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    session_id: Optional[str] = None

class EventStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()

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

    def on_llm_new_token(self, token: str, **kwargs):
        asyncio.create_task(self.queue.put(token))

    def on_llm_end(self, response: LLMResult, **kwargs):
        asyncio.create_task(self.queue.put(None))

    def on_llm_error(self, error: Exception, **kwargs):
        asyncio.create_task(self.queue.put(f"[ERROR] LLM failed: {str(error)}"))

@sleep_and_retry
@limits(calls=100, period=60)
def classify_mode(question: str) -> str:
    prompt = PromptTemplate.from_template(
        "You are a classification bot.\n"
        "Classify the intent of this question: \"{question}\"\n"
        "Your output must be one of: qa, summarize, action_items."
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=get_openai_api_key()
    )
    chain = RunnableSequence(prompt | llm)
    try:
        result = chain.invoke({"question": question}).content.strip().lower()
        logger.info(f"[CLASSIFIER] Question: {question}, Mode: {result}")
        return result if result in {"qa", "summarize", "action_items"} else "qa"
    except Exception as e:
        logger.error(f"[CLASSIFIER] Error classifying question: {e}")
        return "qa"

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
    verbose: bool = False
):
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        mode = classify_mode(question)
        if verbose:
            logger.info(f"[BOT] Detected mode: {mode}")

        retriever_service = RetrieverService()
        retriever = retriever_service.build_retriever(
            company_id=company_id,
            bot_id=bot_id,
            k=k,
            similarity_threshold=similarity_threshold
        )

        prompt = get_prompt_template(mode)

        document_prompt = PromptTemplate.from_template(
            "{page_content}\n[source: {file_name}#{chunk_index}]"
        )

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            streaming=streaming,
            callbacks=[stream_handler] if streaming else None,
            openai_api_key=get_openai_api_key()
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            combine_docs_chain_kwargs={
                "prompt": prompt,
                "document_prompt": document_prompt
            },
            return_source_documents=True,
            verbose=verbose,
            output_key="answer"
        )

        chain_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        if streaming:
            async def run_chain():
                try:
                    result = await asyncio.to_thread(
                        chain_with_history.invoke,
                        {"question": question},
                        config={"configurable": {"session_id": session_id}}
                    )
                    sources = result.get("source_documents", [])
                    if sources:
                        citation_text = "\n\nSources:\n" + "\n".join(
                            f"- {doc.metadata.get('file_name', 'Unknown')} (chunk {doc.metadata.get('chunk_index')})"
                            for doc in sources
                        )
                        await stream_handler.queue.put(citation_text)
                        scores = [doc.metadata.get('score', 'N/A') for doc in sources]
                        logger.info(f"[RETRIEVAL] Sources: {[doc.metadata.get('file_name') for doc in sources]}, Scores: {scores}")
                except Exception as e:
                    logger.error(f"[STREAM] Error in chain: {e}")
                    await stream_handler.queue.put(f"[ERROR] {str(e)}")
                finally:
                    await stream_handler.queue.put(None)
            return run_chain()
        else:
            result = await asyncio.to_thread(
                chain_with_history.invoke,
                {"question": question},
                config={"configurable": {"session_id": session_id}}
            )
            sources = result.get("source_documents", [])
            scores = [doc.metadata.get('score', 'N/A') for doc in sources]
            logger.info(f"[RETRIEVAL] Sources: {[doc.metadata.get('file_name') for doc in sources]}, Scores: {scores}")
            return result

    except Exception as e:
        logger.error(f"[BOT] Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/ask")
async def ask_api(request: AskRequest):
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"[API] Received question: {request.question}, company_id: {request.company_id}, bot_id: {request.bot_id}, session_id: {session_id}")
    stream_handler = EventStreamHandler()
    task = ask_question(
        question=request.question,
        company_id=request.company_id,
        bot_id=request.bot_id,
        session_id=session_id,
        streaming=True,
        stream_handler=stream_handler,
        k=request.k,
        similarity_threshold=request.similarity_threshold,
        verbose=True
    )
    asyncio.create_task(task)
    return StreamingResponse(
        stream_handler.astream(),
        media_type="text/event-stream",
        headers={"X-Session-ID": session_id}
    )

if __name__ == "__main__":
    logger.info("[START] Conversational Mode Enabled. Type 'exit' to quit.")
    company_id = input("🏢 Enter company_id: ").strip()
    bot_id = input("🤖 Enter bot_id: ").strip()
    session_id = str(uuid.uuid4())
    logger.info(f"[CLI] Started session {session_id}")

    while True:
        question = input("\n🧠 You: ")
        if question.lower() in ["exit", "quit"]:
            session_histories.pop(session_id, None)
            logger.info(f"[CLI] Ended session {session_id}")
            break

        try:
            result = asyncio.run(ask_question(
                question=question,
                company_id=int(company_id),
                bot_id=int(bot_id),
                session_id=session_id,
                streaming=False,
                k=None,
                similarity_threshold=None,
                verbose=True
            ))

            logger.info(f"[BOT] Answer: {result['answer']}")
            print("\n🤖 Bot:", result["answer"])
            sources = result.get("source_documents", [])
            if sources:
                print("\n📎 Sources:")
                for doc in sources:
                    name = doc.metadata.get("file_name", "Unknown")
                    chunk = doc.metadata.get("chunk_index", "?")
                    print(f"- {name} (chunk {chunk})")

        except Exception as e:
            logger.error(f"[CLI] Error: {e}")
            print(f"\n❌ Error: {e}")
