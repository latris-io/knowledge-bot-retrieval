import os
import logging
import json
import urllib.parse
import re
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document as CoreDocument

from chromadb import HttpClient
from chromadb.config import Settings

from bot_config import get_openai_api_key
from ratelimit import limits, sleep_and_retry

logger = logging.getLogger(__name__)

DEFAULT_K = int(os.getenv("RETRIEVER_K", 15))  # Increased for better coverage
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVER_SIMILARITY_THRESHOLD", 0.05))  # Lowered for broader matching


class QueryPattern:
    """Store learned patterns for query-document relationships"""
    def __init__(self):
        self.patterns = defaultdict(lambda: defaultdict(float))
        self.pattern_counts = defaultdict(int)
        self.last_updated = datetime.now()
    
    def update_pattern(self, query_type: str, doc_structure: str, success_score: float):
        """Update pattern weights based on retrieval success"""
        self.patterns[query_type][doc_structure] += success_score
        self.pattern_counts[query_type] += 1
        self.last_updated = datetime.now()
    
    def get_pattern_weight(self, query_type: str, doc_structure: str) -> float:
        """Get learned weight for query-document pattern"""
        if query_type not in self.patterns:
            return 1.0
        
        total_weight = sum(self.patterns[query_type].values())
        if total_weight == 0:
            return 1.0
            
        return self.patterns[query_type][doc_structure] / total_weight


class ContentAgnosticRetriever:
    """Enhanced retriever with content-agnostic improvements"""
    
    def __init__(self, embedding_function, llm):
        self.embedding_function = embedding_function
        self.llm = llm
        self.query_patterns = QueryPattern()
        self.entity_cache = {}
        self.concept_cache = {}
        
    def extract_entities(self, query: str) -> List[str]:
        """Extract entities from query using simple heuristics"""
        if query in self.entity_cache:
            return self.entity_cache[query]
        
        # Simple entity extraction - proper nouns, capitalized words, quoted terms
        entities = []
        
        # Find quoted terms
        quoted_terms = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_terms)
        
        # Find capitalized words (likely names, technologies, etc.)
        capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        entities.extend(capitalized_words)
        
        # Find technology-like terms (contains numbers, dots, or common tech patterns)
        tech_terms = re.findall(r'\b[a-zA-Z]+\d+[a-zA-Z]*\b|\b[a-zA-Z]+\.[a-zA-Z]+\b', query)
        entities.extend(tech_terms)
        
        # Find potential names and significant terms (even if lowercase)
        # Look for words that might be names or important terms
        words = re.findall(r'\b[a-zA-Z]+\b', query)
        for word in words:
            # Skip very common words
            if word.lower() in {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about', 'have', 'has', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'cannot', 'what', 'when', 'where', 'who', 'why', 'how', 'experience', 'skilled', 'know', 'familiar', 'worked', 'used'}:
                continue
            
            # Include words that look like names or important terms
            if len(word) > 2:
                # Check if it's a potential name (starts with common name patterns)
                # or if it's a compound word, or if it's longer than 4 characters (likely specific)
                if (len(word) > 4 or 
                    re.match(r'^[a-zA-Z]*[bcdfghjklmnpqrstvwxyz][aeiou]', word.lower()) or
                    re.search(r'[aeiou][bcdfghjklmnpqrstvwxyz][aeiou]', word.lower()) or
                    word.lower().endswith('soft') or word.lower().endswith('force') or
                    word.lower().endswith('ware') or word.lower().endswith('tech') or
                    word.lower().endswith('sys') or word.lower().endswith('log') or
                    word.lower().endswith('base') or word.lower().endswith('hub') or
                    word.lower().endswith('lab') or word.lower().endswith('labs') or
                    word.lower().endswith('corp') or word.lower().endswith('inc') or
                    word.lower().endswith('ltd') or word.lower().endswith('llc')):
                    entities.append(word)
        
        # Remove duplicates and common words
        entities = list(set([e.lower() for e in entities if len(e) > 2]))
        
        self.entity_cache[query] = entities
        return entities
    
    def extract_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query"""
        if query in self.concept_cache:
            return self.concept_cache[query]
        
        # Remove stop words and extract meaningful concepts
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'cannot'}
        
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        concepts = [word for word in words if word not in stop_words and len(word) > 2]
        
        self.concept_cache[query] = concepts
        return concepts
    
    def classify_query_type(self, query: str) -> str:
        """Classify query type for adaptive processing"""
        query_lower = query.lower()
        
        # Relationship queries (does X have Y, is X related to Y)
        if re.search(r'\b(does|do|did|has|have|had|is|are|was|were)\b.*\b(have|has|know|experience|skilled|familiar|worked|used|related|connected)\b', query_lower):
            return 'relationship'
        
        # Factual queries (what is X, when did Y happen)
        if re.search(r'\b(what|when|where|who|which|why|how)\b', query_lower):
            return 'factual'
        
        # Comparison queries (compare X and Y, X vs Y)
        if re.search(r'\b(compare|comparison|versus|vs|difference|different|similar|alike)\b', query_lower):
            return 'comparison'
        
        # List queries (list all X, show me Y)
        if re.search(r'\b(list|all|every|each|show|display|enumerate)\b', query_lower):
            return 'list'
        
        return 'general'
    
    async def expand_query_semantically(self, query: str) -> List[str]:
        """Generate semantic alternatives for the query"""
        try:
            expansion_prompt = f"""Given this query: "{query}"

Generate 3 alternative ways to phrase this same question that might appear in documents.
Consider different vocabulary, document structures, and professional contexts.
Focus on how this information might be presented in resumes, documents, or professional profiles.

Return only the alternative phrasings, one per line, without numbers or bullets."""
            
            # Use invoke instead of agenerate for more reliable results
            response = await self.llm.ainvoke(expansion_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            alternatives = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
            
            # Limit to 3 alternatives to avoid overwhelming the system
            alternatives = alternatives[:3]
            
            logger.info(f"[ENHANCED_RETRIEVER] Generated {len(alternatives)} query alternatives")
            return [query] + alternatives
            
        except Exception as e:
            logger.warning(f"[ENHANCED_RETRIEVER] Query expansion failed: {e}")
            return [query]
    
    def get_adaptive_similarity_threshold(self, query: str) -> float:
        """Adjust similarity threshold based on query characteristics"""
        query_type = self.classify_query_type(query)
        entity_count = len(self.extract_entities(query))
        concept_count = len(self.extract_concepts(query))
        query_length = len(query.split())
        
        # Base threshold
        base_threshold = DEFAULT_SIMILARITY_THRESHOLD
        
        # Adjust based on query type
        if query_type == 'relationship':
            # Relationship queries need broader matching
            base_threshold *= 0.6  # Lower threshold for broader matching
        elif query_type == 'factual':
            # Factual queries can be more precise
            base_threshold *= 1.2  # Higher threshold for precision
        elif query_type == 'comparison':
            # Comparison queries need comprehensive results
            base_threshold *= 0.8  # Slightly lower threshold
        
        # Adjust based on complexity
        if entity_count > 2 or concept_count > 4:
            # Complex queries with many entities/concepts
            base_threshold *= 0.8  # Lower threshold for complex queries
        
        if query_length > 10:
            # Long queries might need more flexible matching
            base_threshold *= 0.9
        
        # Ensure reasonable bounds
        return max(0.01, min(0.1, base_threshold))
    
    def create_contextual_embeddings(self, document_chunk: str, metadata: dict) -> np.ndarray:
        """Create embeddings with document structure context"""
        context_prefix = ""
        
        # Add structure context
        structure_type = metadata.get('structure_type', '')
        if structure_type == 'header':
            context_prefix = "Section heading: "
        elif structure_type == 'paragraph':
            context_prefix = "Content: "
        elif structure_type == 'table_row':
            context_prefix = "Table data: "
        elif structure_type == 'overview':
            context_prefix = "Document overview: "
        
        # Add document context
        file_name = metadata.get('file_name', '')
        if file_name:
            if 'resume' in file_name.lower() or 'cv' in file_name.lower():
                context_prefix += "Professional profile information: "
            elif 'newsletter' in file_name.lower():
                context_prefix += "Newsletter content: "
            elif 'office' in file_name.lower():
                context_prefix += "Office information: "
        
        # Add chunk context
        chunk_index = metadata.get('chunk_index', 0)
        if chunk_index > 0:
            context_prefix += f"Following document section {chunk_index}: "
        
        enhanced_text = context_prefix + document_chunk
        return self.embedding_function.embed_query(enhanced_text)
    
    def calculate_semantic_similarity(self, query: str, document: str) -> float:
        """Calculate semantic similarity between query and document"""
        try:
            query_embedding = self.embedding_function.embed_query(query)
            doc_embedding = self.embedding_function.embed_query(document)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            return float(similarity)
        except Exception as e:
            logger.warning(f"[ENHANCED_RETRIEVER] Similarity calculation failed: {e}")
            return 0.0
    
    def learn_document_patterns(self, query: str, retrieved_docs: List[Document], success_score: float):
        """Learn patterns from successful retrievals"""
        query_type = self.classify_query_type(query)
        
        for doc in retrieved_docs:
            metadata = doc.metadata
            doc_structure = metadata.get('structure_type', 'unknown')
            
            # Update pattern weights
            self.query_patterns.update_pattern(query_type, doc_structure, success_score)
    
    def reweight_results(self, query: str, retrieved_docs: List[Document]) -> List[Document]:
        """Reweight results based on learned patterns"""
        query_type = self.classify_query_type(query)
        
        # Apply learned weights
        for doc in retrieved_docs:
            metadata = doc.metadata
            doc_structure = metadata.get('structure_type', 'unknown')
            
            # Get learned weight for this query-document pattern
            pattern_weight = self.query_patterns.get_pattern_weight(query_type, doc_structure)
            
            # Apply weight to document (store in metadata for downstream use)
            doc.metadata['pattern_weight'] = pattern_weight
        
        # Sort by pattern weight (descending)
        weighted_docs = sorted(retrieved_docs, key=lambda d: d.metadata.get('pattern_weight', 1.0), reverse=True)
        
        return weighted_docs
    
    async def multi_vector_search(self, query: str, vectorstore, k: int = 5) -> List[Document]:
        """Search using multiple vector approaches"""
        all_results = []
        
        # 1. Original query search
        original_results = vectorstore.similarity_search(query, k=k)
        all_results.extend(original_results)
        
        # 2. Entity-focused search
        entities = self.extract_entities(query)
        if entities:
            entity_query = " ".join(entities)
            entity_results = vectorstore.similarity_search(entity_query, k=k)
            all_results.extend(entity_results)
        
        # 3. Concept-focused search
        concepts = self.extract_concepts(query)
        if concepts:
            concept_query = " ".join(concepts)
            concept_results = vectorstore.similarity_search(concept_query, k=k)
            all_results.extend(concept_results)
        
        # 4. Semantic query expansion search
        expanded_queries = await self.expand_query_semantically(query)
        for expanded_query in expanded_queries[1:]:  # Skip original query
            expanded_results = vectorstore.similarity_search(expanded_query, k=k//2)
            all_results.extend(expanded_results)
        
        # Deduplicate results while preserving order
        seen = set()
        unique_results = []
        for doc in all_results:
            doc_id = doc.page_content[:100]  # Use first 100 chars as ID
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
        
        # Return top k results
        return unique_results[:k]
    
    async def hierarchical_search(self, query: str, vectorstore, k: int = 12) -> List[Document]:
        """Hierarchical search: broad entity search then focused refinement"""
        # Step 1: Broad entity search
        entities = self.extract_entities(query)
        broad_results = []
        
        for entity in entities:
            entity_results = vectorstore.similarity_search(entity, k=k*2)
            broad_results.extend(entity_results)
        
        # If no entities found, use concept search
        if not broad_results:
            concepts = self.extract_concepts(query)
            for concept in concepts:
                concept_results = vectorstore.similarity_search(concept, k=k*2)
                broad_results.extend(concept_results)
        
        # Step 2: Focused search within broad results
        threshold = self.get_adaptive_similarity_threshold(query)
        focused_results = []
        
        for result in broad_results:
            relevance_score = self.calculate_semantic_similarity(query, result.page_content)
            if relevance_score > threshold:
                result.metadata['relevance_score'] = relevance_score
                focused_results.append(result)
        
        # Sort by relevance and deduplicate
        focused_results.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        
        # Deduplicate
        seen = set()
        unique_results = []
        for doc in focused_results:
            doc_id = doc.page_content[:100]
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
        
        return unique_results[:k]


class EnhancedCustomRetriever(BaseRetriever):
    """Custom retriever that returns pre-computed enhanced results"""
    
    def __init__(self, results: List[Document]):
        super().__init__()
        self._results = results
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Required method for BaseRetriever"""
        return self._results
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Legacy method for compatibility"""
        return self._results


class RetrieverService:
    def __init__(self):
        self.embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=get_openai_api_key()
        )
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=get_openai_api_key()
        )
        self.enhanced_retriever = ContentAgnosticRetriever(self.embedding_function, self.llm)
        logger.info("[RETRIEVER] Initialized RetrieverService with enhanced capabilities")

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
    async def build_enhanced_retriever(
        self,
        company_id: int,
        bot_id: int,
        query: str,
        filters: Optional[Dict] = None,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_multi_query: bool = False,
        use_enhanced_search: bool = True
    ):
        """Build retriever with enhanced content-agnostic capabilities"""
        try:
            collection_name = "global"
            vectorstore = self.get_chroma_vectorstore(collection_name)
            logger.info(f"[ENHANCED_RETRIEVER] Connected to Chroma collection: {collection_name}")

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
            
            # Use adaptive similarity threshold
            if similarity_threshold is None:
                similarity_threshold = self.enhanced_retriever.get_adaptive_similarity_threshold(query)
            
            logger.info(f"[ENHANCED_RETRIEVER] Using adaptive similarity threshold: {similarity_threshold}")

            # Get all documents for BM25
            docs = vectorstore.get(include=["documents", "metadatas"], where=base_filter)
            texts = docs["documents"]
            metadatas = docs["metadatas"]

            if use_enhanced_search:
                # Use enhanced search strategies
                if use_multi_query:
                    # Multi-vector search with query expansion
                    logger.info("[ENHANCED_RETRIEVER] Using multi-vector search with query expansion")
                    enhanced_results = await self.enhanced_retriever.multi_vector_search(query, vectorstore, k)
                else:
                    # Hierarchical search
                    logger.info("[ENHANCED_RETRIEVER] Using hierarchical search")
                    enhanced_results = await self.enhanced_retriever.hierarchical_search(query, vectorstore, k)
                
                # Apply learned patterns
                enhanced_results = self.enhanced_retriever.reweight_results(query, enhanced_results)
                
                # Return enhanced custom retriever with pre-computed results
                return EnhancedCustomRetriever(enhanced_results)
            
            else:
                # Fall back to original retriever logic
                return await self.build_original_retriever(
                    company_id, bot_id, filters, k, similarity_threshold, use_multi_query
                )

        except Exception as e:
            logger.error(f"[ENHANCED_RETRIEVER] Error building enhanced retriever: {e}", exc_info=True)
            # Fall back to original retriever
            return await self.build_original_retriever(
                company_id, bot_id, filters, k, similarity_threshold, use_multi_query
            )

    async def build_original_retriever(
        self,
        company_id: int,
        bot_id: int,
        filters: Optional[Dict] = None,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_multi_query: bool = False
    ):
        """Original retriever logic as fallback"""
        collection_name = "global"
        vectorstore = self.get_chroma_vectorstore(collection_name)

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
            query_prompt = PromptTemplate(
                input_variables=["question"],
                template="""Generate 2 alternative search queries for the following question to improve retrieval coverage.
Focus on different aspects and phrasings.
Question: {question}
Alternative queries:"""
            )
            
            multi_query = MultiQueryRetriever.from_llm(
                retriever=vector_retriever,
                llm=self.llm,
                prompt=query_prompt
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

    # Keep the original build_retriever method for backward compatibility
    def build_retriever(
        self,
        company_id: int,
        bot_id: int,
        filters: Optional[Dict] = None,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_multi_query: bool = False
    ):
        """Backward compatibility method - calls original retriever"""
        import asyncio
        return asyncio.run(self.build_original_retriever(
            company_id, bot_id, filters, k, similarity_threshold, use_multi_query
        ))
