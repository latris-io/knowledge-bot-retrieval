import os
import logging
import json
import urllib.parse
from typing import Dict, Optional

from langchain_openai import OpenAIEmbeddings
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

DEFAULT_K = int(os.getenv("RETRIEVER_K", 12))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVER_SIMILARITY_THRESHOLD", 0.1))


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
        similarity_threshold: Optional[float] = None
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

            # Use direct vector retriever for maximum speed (skip multi-query overhead)
            docs = vectorstore.get(include=["documents", "metadatas"], where=base_filter)
            texts = docs["documents"]
            metadatas = docs["metadatas"]

            bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
            bm25.k = k
            logger.info(f"[RETRIEVER] Initialized BM25 with {len(texts)} documents")

            hybrid = EnsembleRetriever(
                retrievers=[vector_retriever, bm25],
                weights=[0.8, 0.2]
            )

            logger.info(f"[RETRIEVER DEBUG] EmbeddingsFilter threshold set to {similarity_threshold}")
            compressor = DocumentCompressorPipeline(transformers=[
                EmbeddingsFilter(
                    embeddings=self.embedding_function,
                    similarity_threshold=similarity_threshold
                )
            ])

            reranked = ContextualCompressionRetriever(
                base_retriever=hybrid,
                base_compressor=compressor
            )

            logger.info(f"[RETRIEVER] Hybrid retriever ready — k={k}, threshold={similarity_threshold}")

            if metadatas:
                sample_meta = metadatas[0]
                logger.info(f"[RETRIEVER DEBUG] Sample document metadata: {json.dumps(sample_meta, indent=2)}")

            return reranked

        except Exception as e:
            logger.error(f"[RETRIEVER] Error building retriever: {e}", exc_info=True)
            raise
