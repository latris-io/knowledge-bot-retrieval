import os
import logging
import json
import urllib.parse
from typing import Dict, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain_community.vectorstores import Chroma

from chromadb import HttpClient
from chromadb.config import Settings

from bot_config import get_openai_api_key
from ratelimit import limits, sleep_and_retry

logger = logging.getLogger(__name__)

DEFAULT_K = int(os.getenv("RETRIEVER_K", 15))  # Increased for better coverage
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVER_SIMILARITY_THRESHOLD", 0.05))  # Lowered for broader matching


class RetrieverService:
    def __init__(self):
        self.embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=get_openai_api_key()
        )
        logger.info("[RETRIEVER] Initialized RetrieverService")

    def get_chroma_vectorstore(self, collection_name: str):
        chroma_url = os.getenv("CHROMA_URL")
        if not chroma_url:
            raise ValueError("CHROMA_URL must be set in environment")

        parsed = urllib.parse.urlparse(chroma_url)
        client = HttpClient(
            host=parsed.hostname,
            port=parsed.port or 443,
            ssl=parsed.scheme == "https",
            settings=Settings(anonymized_telemetry=False)
        )

        collection = client.get_or_create_collection(name=collection_name)

        return Chroma(
            collection_name=collection_name,
            client=client,
            embedding_function=self.embedding_function
        )

    @sleep_and_retry
    @limits(calls=100, period=60)
    def build_retriever(
        self,
        company_id: int,
        bot_id: int,
        filters: Optional[Dict] = None,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_multi_query: bool = False
    ):
        try:
            collection_name = "global"
            vectorstore = self.get_chroma_vectorstore(collection_name)
            logger.info(f"[RETRIEVER] Connected to Chroma collection: {collection_name}")

            base_filter = {
                "$and": [
                    {"company_id": {"$eq": company_id}},
                    {"bot_id": {"$eq": bot_id}}
                ]
            }
            if filters:
                for key, value in filters.items():
                    base_filter["$and"].append({key: {"$eq": value}})

            k = k if k is not None else DEFAULT_K
            similarity_threshold = similarity_threshold if similarity_threshold is not None else DEFAULT_SIMILARITY_THRESHOLD

            vector_retriever = vectorstore.as_retriever(
                search_kwargs={"k": k, "filter": base_filter}
            )

            # Conditionally use MultiQueryRetriever for enhanced coverage vs speed
            if use_multi_query:
                logger.info(f"[RETRIEVER] Using optimized MultiQueryRetriever for enhanced coverage")
                # Custom prompt for faster query generation
                from langchain.prompts import PromptTemplate
                
                query_prompt = PromptTemplate(
                    input_variables=["question"],
                    template="""Generate 2 alternative search queries for the following question to improve retrieval coverage.
Focus on different aspects and phrasings.
Question: {question}
Alternative queries:"""
                )
                
                multi_query = MultiQueryRetriever.from_llm(
                    retriever=vector_retriever,
                    llm=ChatOpenAI(
                        model="gpt-3.5-turbo",  # Faster model for query generation
                        temperature=0,
                        openai_api_key=get_openai_api_key()
                    ),
                    prompt=query_prompt  # Custom prompt for 2 queries instead of 3
                )
                vector_component = multi_query
            else:
                logger.info(f"[RETRIEVER] Using direct vector retriever for maximum speed")
                vector_component = vector_retriever

            docs = vectorstore.get(include=["documents", "metadatas"], where=base_filter)
            texts = docs["documents"]
            metadatas = docs["metadatas"]

            bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
            bm25.k = k
            logger.info(f"[RETRIEVER] Initialized BM25 with {len(texts)} documents")

            # For maximum speed in direct mode, skip redundant embedding compression
            if use_multi_query:
                # Full hybrid with compression for comprehensive queries
                hybrid = EnsembleRetriever(
                    retrievers=[vector_component, bm25],
                    weights=[0.7, 0.3]  # Adjusted to enhance keyword matching for multi-query scenarios
                )

                # Use relaxed threshold for complex queries to speed up filtering
                complex_threshold = max(similarity_threshold * 0.7, 0.05)  # More permissive filtering
                logger.info(f"[RETRIEVER DEBUG] EmbeddingsFilter threshold set to {complex_threshold} (relaxed for complex queries)")
                compressor = DocumentCompressorPipeline(transformers=[
                    EmbeddingsFilter(
                        embeddings=self.embedding_function,
                        similarity_threshold=complex_threshold
                    )
                ])

                reranked = ContextualCompressionRetriever(
                    base_retriever=hybrid,
                    base_compressor=compressor
                )
                
                logger.info(f"[RETRIEVER] Full hybrid retriever with compression — k={k}, threshold={similarity_threshold}")
                final_retriever = reranked
            else:
                # Adaptive direct mode: vector-only for simple/medium, hybrid for complex
                if k >= 8:
                    # High-k complex queries: use BM25+Vector hybrid for comprehensive coverage
                    hybrid = EnsembleRetriever(
                        retrievers=[vector_component, bm25],
                        weights=[0.6, 0.4]  # Adjusted to favor keyword matching for improved retrieval accuracy
                    )
                    logger.info(f"[RETRIEVER] Fast comprehensive hybrid retriever — k={k} (BM25+Vector)")
                    final_retriever = hybrid
                else:
                    # Simple/medium queries: vector-only for maximum speed
                    logger.info(f"[RETRIEVER] Direct vector-only retriever for maximum speed — k={k}")
                    final_retriever = vector_component

            if metadatas:
                sample_meta = metadatas[0]
                logger.info(f"[RETRIEVER DEBUG] Sample document metadata: {json.dumps(sample_meta, indent=2)}")

            # Enhanced debug logging for retrieval testing
            logger.info(f"[RETRIEVER DEBUG] Configuration: k={k}, similarity_threshold={similarity_threshold}")
            logger.info(f"[RETRIEVER DEBUG] Total documents in corpus: {len(texts)}")
            logger.info(f"[RETRIEVER DEBUG] Using {'MultiQuery' if use_multi_query else 'Direct'} retrieval strategy")

            return final_retriever

        except Exception as e:
            logger.error(f"[RETRIEVER] Error building retriever: {e}", exc_info=True)
            raise
