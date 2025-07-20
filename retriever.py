import os
import logging
import json
import urllib.parse
from typing import Dict, Optional
import asyncio

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

DEFAULT_K = int(os.getenv("RETRIEVER_K", 12))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVER_SIMILARITY_THRESHOLD", 0.05))  # FI-01: Lowered from 0.1 to 0.05 for broader matching


class RetrieverService:
    def __init__(self):
        self.embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=get_openai_api_key()
        )
        # Cache expensive objects
        self._vectorstore_cache = {}
        self._bm25_cache = {}
        # FI-04: Enhanced Retrieval System
        self._query_cache = {}  # Cache for query expansions
        logger.info("[RETRIEVER] Initialized RetrieverService")

    def get_chroma_vectorstore(self, collection_name: str):
        # Cache vectorstore to avoid recreating ChromaDB connections
        if collection_name in self._vectorstore_cache:
            logger.info(f"[RETRIEVER] Using cached vectorstore for {collection_name}")
            return self._vectorstore_cache[collection_name]
            
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

        vectorstore = Chroma(
            collection_name=collection_name,
            client=client,
            embedding_function=self.embedding_function
        )
        self._vectorstore_cache[collection_name] = vectorstore
        return vectorstore

    # FI-04: Content-Agnostic Enhanced Retrieval System
    async def expand_query_semantically(self, query: str) -> list:
        """Generate alternative query formulations for better retrieval coverage"""
        if query in self._query_cache:
            return self._query_cache[query]
            
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,  # Slight creativity for alternatives
                openai_api_key=get_openai_api_key()
            )
            
            # Content-agnostic query expansion prompt
            expansion_prompt = f"""Generate 2 alternative search queries that would help find the same information as: "{query}"

Focus on:
- Different vocabulary and phrasings
- Various ways to ask the same question  
- Alternative technical terms or synonyms

Original: {query}
Alternative 1:
Alternative 2:"""

            response = await asyncio.to_thread(llm.invoke, expansion_prompt)
            alternatives = [line.strip() for line in response.content.split('\n') if line.strip() and not line.startswith('Alternative')]
            
            # Always include original query
            expanded_queries = [query] + [alt for alt in alternatives if alt and len(alt) > 5][:2]
            
            self._query_cache[query] = expanded_queries
            logger.info(f"[FI-04] Expanded query into {len(expanded_queries)} variations")
            return expanded_queries
            
        except Exception as e:
            logger.error(f"[FI-04] Query expansion failed: {e}")
            return [query]  # Fallback to original

    async def multi_vector_search(self, query: str, vectorstore, k: int = 8) -> list:
        """FI-04: Enhanced multi-vector search with query expansion"""
        try:
            # Get query expansions
            expanded_queries = await self.expand_query_semantically(query)
            
            all_results = []
            seen_docs = set()
            
            # Search with each query variation
            for i, variant_query in enumerate(expanded_queries):
                try:
                    # Different search approaches for each variant
                    if i == 0:
                        # Original query - standard search
                        results = vectorstore.similarity_search(variant_query, k=k//2)
                    elif i == 1:
                        # First alternative - entity focused
                        results = vectorstore.similarity_search(variant_query, k=k//3)
                    else:
                        # Additional alternatives - concept focused
                        results = vectorstore.similarity_search(variant_query, k=k//4)
                    
                    # Deduplicate while preserving order
                    for doc in results:
                        doc_key = f"{doc.page_content[:100]}_{doc.metadata.get('chunk_index', 0)}"
                        if doc_key not in seen_docs:
                            seen_docs.add(doc_key)
                            doc.metadata['query_variant'] = i
                            doc.metadata['search_approach'] = ['original', 'entity_focused', 'concept_focused'][min(i, 2)]
                            all_results.append(doc)
                    
                except Exception as e:
                    logger.error(f"[FI-04] Search variant {i} failed: {e}")
                    continue
            
            # Limit to requested k with quality ranking
            final_results = all_results[:k]
            logger.info(f"[FI-04] Multi-vector search: {len(expanded_queries)} queries → {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"[FI-04] Multi-vector search failed: {e}")
            # Fallback to standard search
            return vectorstore.similarity_search(query, k=k)

    # FI-05: Content-Agnostic Semantic Bias Fix
    def analyze_query_term_importance(self, query: str) -> dict:
        """Analyze query terms for importance without domain-specific patterns"""
        import re
        
        # Extract meaningful terms (remove stop words)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 
            'where', 'who', 'why', 'how', 'this', 'that', 'these', 'those'
        }
        
        # Clean and extract terms
        terms = re.findall(r'\b\w+\b', query.lower())
        meaningful_terms = [term for term in terms if term not in stop_words and len(term) > 2]
        
        importance_scores = {}
        
        for term in meaningful_terms:
            importance = 1.0
            
            # FI-05: Universal importance heuristics (no domain-specific patterns)
            
            # Length-based scoring - longer terms often more specific
            if len(term) >= 6:
                importance *= 1.5
            elif len(term) >= 4:
                importance *= 1.2
                
            # Capitalization detection in original query - proper nouns important
            if term.title() in query or term.upper() in query:
                importance *= 1.4
                
            # Position-based importance - terms at start/end often key
            original_terms = re.findall(r'\b\w+\b', query)
            if term.lower() in [original_terms[0].lower(), original_terms[-1].lower()] if original_terms else []:
                importance *= 1.1
                
            # Frequency in query - repeated terms indicate importance
            term_frequency = meaningful_terms.count(term)
            if term_frequency > 1:
                importance *= (1.0 + (term_frequency - 1) * 0.3)
            
            importance_scores[term] = importance
        
        logger.debug(f"[FI-05] Term importance: {importance_scores}")
        return importance_scores

    def rerank_by_term_importance(self, query: str, documents: list) -> list:
        """FI-05: Re-rank documents based on important query terms"""
        if not documents:
            return documents
            
        try:
            term_importance = self.analyze_query_term_importance(query)
            
            if not term_importance:
                return documents
            
            # Calculate importance-based scores for each document
            scored_docs = []
            
            for doc in documents:
                content = doc.page_content.lower()
                importance_score = 0.0
                
                # Score based on presence of important terms
                for term, importance in term_importance.items():
                    term_count = content.count(term.lower())
                    if term_count > 0:
                        # Boost score based on term importance and frequency
                        importance_score += importance * min(term_count, 3)  # Cap at 3 to avoid over-weighting
                
                # Normalize by document length to avoid bias toward long documents
                doc_length = len(content.split())
                if doc_length > 0:
                    importance_score = importance_score / (doc_length / 100)  # Normalize per 100 words
                
                # Store original metadata and add importance score
                doc.metadata['importance_score'] = importance_score
                doc.metadata['term_matches'] = sum(1 for term in term_importance.keys() if term.lower() in content)
                
                scored_docs.append((doc, importance_score))
            
            # Sort by importance score (highest first), then by original order as tiebreaker
            scored_docs.sort(key=lambda x: (-x[1], documents.index(x[0])))
            reranked_docs = [doc for doc, score in scored_docs]
            
            logger.info(f"[FI-05] Re-ranked {len(documents)} docs by term importance")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"[FI-05] Re-ranking failed: {e}")
            return documents  # Fallback to original

    # FI-08: Enhanced Retrieval Quality Improvements
    def calculate_shannon_entropy(self, text: str) -> float:
        """Calculate Shannon entropy for text quality assessment"""
        import math
        from collections import Counter
        
        if not text or len(text) < 10:
            return 0.0
            
        # Calculate character-level entropy
        char_counts = Counter(text.lower())
        total_chars = len(text)
        
        entropy = 0.0
        for count in char_counts.values():
            probability = count / total_chars
            entropy -= probability * math.log2(probability)
            
        return entropy

    def calculate_information_density(self, text: str) -> float:
        """Calculate information density using various metrics"""
        if not text or len(text) < 10:
            return 0.0
        
        # Unique word ratio
        words = text.lower().split()
        unique_words = len(set(words))
        total_words = len(words)
        word_diversity = unique_words / total_words if total_words > 0 else 0
        
        # Average word length (longer words typically more informative)
        avg_word_length = sum(len(word) for word in words) / total_words if total_words > 0 else 0
        word_complexity = min(avg_word_length / 6, 1.0)  # Normalize to 0-1
        
        # Sentence structure (proper punctuation indicates quality)
        punctuation_count = sum(1 for char in text if char in '.!?;:')
        sentence_structure = min(punctuation_count / (len(text) / 100), 1.0)  # Per 100 chars
        
        # Combined information density score
        density = (word_diversity * 0.4 + word_complexity * 0.3 + sentence_structure * 0.3)
        return density

    def filter_by_quality(self, documents: list, min_entropy: float = 3.0, min_density: float = 0.3) -> list:
        """FI-08: Filter documents based on information quality"""
        if not documents:
            return documents
            
        try:
            quality_docs = []
            
            for doc in documents:
                content = doc.page_content
                
                # Skip very short documents
                if len(content) < 20:
                    continue
                
                # Calculate quality metrics
                entropy = self.calculate_shannon_entropy(content)
                density = self.calculate_information_density(content)
                
                # Quality thresholds
                passes_entropy = entropy >= min_entropy
                passes_density = density >= min_density
                passes_length = len(content.split()) >= 5  # Minimum word count
                
                # Store quality metrics in metadata
                doc.metadata['shannon_entropy'] = entropy
                doc.metadata['information_density'] = density
                doc.metadata['quality_score'] = (entropy / 6.0) * 0.6 + density * 0.4  # Normalize and combine
                
                # Apply quality filters
                if passes_entropy and passes_density and passes_length:
                    quality_docs.append(doc)
                else:
                    logger.debug(f"[FI-08] Filtered low-quality doc: entropy={entropy:.2f}, density={density:.2f}")
            
            logger.info(f"[FI-08] Quality filter: {len(documents)} → {len(quality_docs)} docs")
            return quality_docs
            
        except Exception as e:
            logger.error(f"[FI-08] Quality filtering failed: {e}")
            return documents  # Fallback to original

    def smart_deduplicate(self, documents: list, similarity_threshold: float = 0.85) -> list:
        """FI-08: Smart deduplication based on content similarity"""
        if not documents or len(documents) <= 1:
            return documents
            
        try:
            from difflib import SequenceMatcher
            
            unique_docs = []
            seen_contents = []
            
            for doc in documents:
                content = doc.page_content.strip()
                
                # Check similarity with existing documents
                is_duplicate = False
                best_similarity = 0.0
                
                for existing_content in seen_contents:
                    similarity = SequenceMatcher(None, content, existing_content).ratio()
                    best_similarity = max(best_similarity, similarity)
                    
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_docs.append(doc)
                    seen_contents.append(content)
                    doc.metadata['dedup_similarity'] = best_similarity
                else:
                    logger.debug(f"[FI-08] Deduplicated similar content: {best_similarity:.3f} similarity")
            
            logger.info(f"[FI-08] Smart deduplication: {len(documents)} → {len(unique_docs)} docs")
            return unique_docs
            
        except Exception as e:
            logger.error(f"[FI-08] Smart deduplication failed: {e}")
            return documents  # Fallback to original

    def apply_quality_enhancements(self, query: str, documents: list) -> list:
        """FI-08: Apply all quality enhancements in sequence"""
        if not documents:
            return documents
        
        try:
            logger.info(f"[FI-08] Applying quality enhancements to {len(documents)} documents")
            
            # Step 1: Filter by information quality
            quality_filtered = self.filter_by_quality(documents)
            
            # Step 2: Smart deduplication
            deduplicated = self.smart_deduplicate(quality_filtered)
            
            # Step 3: Re-rank by term importance (FI-05)
            reranked = self.rerank_by_term_importance(query, deduplicated)
            
            # Step 4: Final quality scoring for sorting
            for i, doc in enumerate(reranked):
                doc.metadata['final_rank'] = i
                doc.metadata['enhancement_applied'] = True
            
            logger.info(f"[FI-08] Quality enhancements complete: {len(documents)} → {len(reranked)} docs")
            return reranked
            
        except Exception as e:
            logger.error(f"[FI-08] Quality enhancements failed: {e}")
            return documents  # Fallback to original

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

            # Cache BM25 retriever to avoid expensive document fetching and rebuilding
            cache_key = f"{company_id}_{bot_id}"
            if cache_key in self._bm25_cache:
                bm25 = self._bm25_cache[cache_key]
                bm25.k = k  # Update k for this request
                logger.info(f"[RETRIEVER] Using cached BM25 with {len(bm25.docs)} documents")
            else:
                docs = vectorstore.get(include=["documents", "metadatas"], where=base_filter)
                texts = docs["documents"]
                metadatas = docs["metadatas"]
                
                bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
                bm25.k = k
                self._bm25_cache[cache_key] = bm25
                logger.info(f"[RETRIEVER] Initialized BM25 with {len(texts)} documents")

            # For maximum speed in direct mode, skip redundant embedding compression
            if use_multi_query:
                # Full hybrid with compression for comprehensive queries - FI-01: Enhanced BM25 weighting
                hybrid = EnsembleRetriever(
                    retrievers=[vector_component, bm25],
                    weights=[0.6, 0.4]  # FI-01: Enhanced BM25 weight (0.6/0.4) for better keyword matching
                )

                # FI-04: Enhanced multi-vector search integration  
                class EnhancedRetriever:
                    def __init__(self, base_retriever, service, query):
                        self.base_retriever = base_retriever
                        self.service = service
                        self.query = query
                    
                    def get_relevant_documents(self, query_text):
                        # Use FI-04 multi-vector search
                        return asyncio.run(self.service.multi_vector_search(query_text, vectorstore, k))
                    
                    def invoke(self, input_text):
                        docs = self.get_relevant_documents(input_text)
                        # Apply FI-08 quality enhancements
                        enhanced_docs = self.service.apply_quality_enhancements(input_text, docs)
                        return enhanced_docs

                # Create enhanced retriever with all improvements
                enhanced_retriever = EnhancedRetriever(hybrid, self, "")
                logger.info(f"[RETRIEVER] Enhanced hybrid retriever with FI-04, FI-05, FI-08 — k={k}")
                final_retriever = enhanced_retriever
            else:
                # Adaptive direct mode: vector-only for simple/medium, hybrid for complex
                # FI-04, FI-05, FI-08: Enhanced retrieval for all queries
                class StandardEnhancedRetriever:
                    def __init__(self, base_retriever, service, use_hybrid):
                        self.base_retriever = base_retriever
                        self.service = service
                        self.use_hybrid = use_hybrid
                    
                    def get_relevant_documents(self, query_text):
                        if self.use_hybrid:
                            # Use FI-04 multi-vector search
                            return asyncio.run(self.service.multi_vector_search(query_text, vectorstore, k))
                        else:
                            # Standard vector search with enhancements
                            docs = vectorstore.similarity_search(query_text, k=k)
                            return self.service.apply_quality_enhancements(query_text, docs)
                    
                    def invoke(self, input_text):
                        return self.get_relevant_documents(input_text)

                if k >= 8:
                    # High-k complex queries: use enhanced hybrid
                    enhanced_retriever = StandardEnhancedRetriever(vector_component, self, use_hybrid=True)
                    logger.info(f"[RETRIEVER] Enhanced hybrid retriever (FI-04, FI-05, FI-08) — k={k}")
                    final_retriever = enhanced_retriever
                else:
                    # Simple/medium queries: enhanced vector-only 
                    enhanced_retriever = StandardEnhancedRetriever(vector_component, self, use_hybrid=False)
                    logger.info(f"[RETRIEVER] Enhanced vector retriever (FI-05, FI-08) — k={k}")
                    final_retriever = enhanced_retriever

            if metadatas:
                sample_meta = metadatas[0]
                logger.info(f"[RETRIEVER DEBUG] Sample document metadata: {json.dumps(sample_meta, indent=2)}")

            return final_retriever

        except Exception as e:
            logger.error(f"[RETRIEVER] Error building retriever: {e}", exc_info=True)
            raise
            
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
        """
        Enhanced retriever with query-adaptive processing and all Foundation Improvements.
        
        Args:
            company_id: Company ID for filtering
            bot_id: Bot ID for filtering  
            query: The search query for adaptive processing
            filters: Additional metadata filters
            k: Number of documents to retrieve
            similarity_threshold: Similarity threshold for filtering
            use_multi_query: Whether to use MultiQueryRetriever
            use_enhanced_search: Enable enhanced search with FI-04, FI-05, FI-08
        """
        try:
            # Start with base retriever
            base_retriever = self.build_retriever(
                company_id=company_id,
                bot_id=bot_id,
                filters=filters,
                k=k,
                similarity_threshold=similarity_threshold,
                use_multi_query=use_multi_query
            )
            
            # If enhanced search is disabled, return base retriever
            if not use_enhanced_search:
                logger.info("[ENHANCED-RETRIEVER] Enhanced search disabled, using base retriever")
                return base_retriever
            
            logger.info("[ENHANCED-RETRIEVER] Building enhanced retriever with query-adaptive processing")
            
            # Enhanced retriever with all Foundation Improvements
            class EnhancedRetrieverWrapper:
                """Wrapper that applies all Foundation Improvements to retrieval"""
                
                def __init__(self, base_retriever, retriever_service, query):
                    self.base_retriever = base_retriever
                    self.retriever_service = retriever_service
                    self.query = query
                    
                def get_relevant_documents(self, query_text):
                    """Enhanced document retrieval with FI-04, FI-05, FI-08"""
                    
                    # Step 1: FI-04 - Content-Agnostic Enhanced Retrieval System
                    expanded_queries = []
                    if hasattr(self.retriever_service, 'expand_query'):
                        # Use existing query expansion if available
                        expanded_queries = self.retriever_service.expand_query(query_text)
                    else:
                        # Simple query expansion as fallback
                        expanded_queries = [query_text]
                        # Add basic query variations
                        words = query_text.lower().split()
                        if len(words) > 1:
                            # Create entity-focused query
                            entities = [w for w in words if len(w) > 3 and w.isalpha()]
                            if entities:
                                expanded_queries.append(" ".join(entities))
                    
                    # Retrieve documents for each query variant
                    all_documents = []
                    for expanded_query in expanded_queries:
                        try:
                            docs = self.base_retriever.get_relevant_documents(expanded_query)
                            all_documents.extend(docs)
                        except Exception as e:
                            logger.warning(f"[FI-04] Query expansion failed for '{expanded_query}': {e}")
                    
                    # Fall back to original query if no expansion results
                    if not all_documents:
                        all_documents = self.base_retriever.get_relevant_documents(query_text)
                    
                    # Step 2: FI-05 - Content-Agnostic Semantic Bias Fix
                    if hasattr(self.retriever_service, 'apply_semantic_bias_fix'):
                        all_documents = self.retriever_service.apply_semantic_bias_fix(
                            all_documents, query_text
                        )
                    
                    # Step 3: FI-08 - Enhanced Retrieval Quality Improvements  
                    if hasattr(self.retriever_service, 'apply_quality_enhancements'):
                        all_documents = self.retriever_service.apply_quality_enhancements(
                            all_documents, query_text
                        )
                    
                    return all_documents
            
            enhanced_retriever = EnhancedRetrieverWrapper(base_retriever, self, query)
            
            logger.info("[ENHANCED-RETRIEVER] Enhanced retriever built with FI-04, FI-05, FI-08 integration")
            return enhanced_retriever
            
        except Exception as e:
            logger.error(f"[ENHANCED-RETRIEVER] Error building enhanced retriever: {e}")
            # Fallback to base retriever on error
            return self.build_retriever(
                company_id=company_id,
                bot_id=bot_id,
                filters=filters,
                k=k,
                similarity_threshold=similarity_threshold,
                use_multi_query=use_multi_query
            )
