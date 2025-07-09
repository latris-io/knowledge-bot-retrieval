import os
import logging
import json
import urllib.parse
import re
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import asyncio
import hashlib
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

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

# Performance optimizations
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
CACHE_EMBEDDINGS = os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true"
BATCH_EMBEDDINGS = os.getenv("BATCH_EMBEDDINGS", "true").lower() == "true"

logger = logging.getLogger(__name__)

DEFAULT_K = int(os.getenv("RETRIEVER_K", 15))  # Increased for better coverage
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVER_SIMILARITY_THRESHOLD", 0.05))  # Lowered for broader matching


class EmbeddingCache:
    """High-performance embedding cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if len(self.cache) >= self.max_size:
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            self.access_times[text_hash] = time.time()
            self._hits += 1
            return self.cache[text_hash]
        self._misses += 1
        return None
    
    def put(self, text: str, embedding: List[float]):
        """Cache embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        self._evict_lru()
        self.cache[text_hash] = embedding
        self.access_times[text_hash] = time.time()
    
    def stats(self) -> Dict:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self.cache)
        }

class MockEmbedding:
    """Mock embedding for development mode"""
    
    @staticmethod
    def embed_query(text: str) -> List[float]:
        """Generate deterministic mock embedding"""
        # Use text hash to generate consistent mock embeddings
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Generate 3072-dimensional embedding (text-embedding-3-large dimension)
        np.random.seed(int(text_hash[:8], 16))  # Seed from hash for consistency
        return np.random.normal(0, 1, 3072).tolist()
    
    @staticmethod
    def embed_documents(texts: List[str]) -> List[List[float]]:
        """Generate deterministic mock embeddings for multiple texts"""
        return [MockEmbedding.embed_query(text) for text in texts]

class OptimizedEmbeddingFunction:
    """Optimized embedding function with caching and batching"""
    
    def __init__(self, original_embedding_function):
        self.original = original_embedding_function
        self.cache = EmbeddingCache() if CACHE_EMBEDDINGS else None
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding with caching"""
        if DEVELOPMENT_MODE:
            return MockEmbedding.embed_query(text)
        
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached
        
        # Get real embedding
        embedding = self.original.embed_query(text)
        
        if self.cache:
            self.cache.put(text, embedding)
        
        return embedding
    
    async def embed_queries_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding processing"""
        if DEVELOPMENT_MODE:
            return MockEmbedding.embed_documents(texts)
        
        if not BATCH_EMBEDDINGS or len(texts) <= 3:
            # For small batches, use individual cached calls
            embeddings = []
            for text in texts:
                embeddings.append(self.embed_query(text))
            return embeddings
        
        # Check cache first
        cached_embeddings = {}
        uncached_texts = []
        
        if self.cache:
            for text in texts:
                cached = self.cache.get(text)
                if cached is not None:
                    cached_embeddings[text] = cached
                else:
                    uncached_texts.append(text)
        else:
            uncached_texts = texts
        
        # Batch process uncached texts
        new_embeddings = {}
        if uncached_texts:
            try:
                # Use batch API if available
                if hasattr(self.original, 'embed_documents'):
                    batch_results = self.original.embed_documents(uncached_texts)
                    for text, embedding in zip(uncached_texts, batch_results):
                        new_embeddings[text] = embedding
                        if self.cache:
                            self.cache.put(text, embedding)
                else:
                    # Parallel individual calls
                    async def embed_single(text):
                        return text, self.original.embed_query(text)
                    
                    tasks = [embed_single(text) for text in uncached_texts]
                    results = await asyncio.gather(*tasks)
                    
                    for text, embedding in results:
                        new_embeddings[text] = embedding
                        if self.cache:
                            self.cache.put(text, embedding)
            except Exception as e:
                logger.warning(f"[EMBEDDING_CACHE] Batch processing failed: {e}")
                # Fall back to individual calls
                for text in uncached_texts:
                    embedding = self.original.embed_query(text)
                    new_embeddings[text] = embedding
                    if self.cache:
                        self.cache.put(text, embedding)
        
        # Combine cached and new embeddings in original order
        result = []
        for text in texts:
            if text in cached_embeddings:
                result.append(cached_embeddings[text])
            else:
                result.append(new_embeddings[text])
        
        return result
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        if self.cache:
            return self.cache.stats()
        return {"cache_disabled": True}


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
        self.embedding_function = OptimizedEmbeddingFunction(embedding_function)
        self.llm = llm
        self.query_patterns = QueryPattern()
        self.entity_cache = {}
        self.concept_cache = {}
        self._connection_pool = {}  # Connection pooling
        
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
        # In development mode, use fast deterministic alternatives
        if DEVELOPMENT_MODE:
            entities = self.extract_entities(query)
            concepts = self.extract_concepts(query)
            
            # Generate deterministic alternatives based on entities and concepts
            alternatives = []
            if entities:
                alternatives.append(" ".join(entities))
            if concepts and len(concepts) >= 2:
                alternatives.append(" ".join(concepts[:2]))
            if len(concepts) > 2:
                alternatives.append(" ".join(concepts[2:4]))
            
            # Pad with variations if needed
            while len(alternatives) < 2:
                alternatives.append(f"professional {query}")
            
            logger.info(f"[ENHANCED_RETRIEVER] Generated {len(alternatives)} mock query alternatives")
            return [query] + alternatives[:2]  # Limit to 2 for speed
        
        try:
            expansion_prompt = f"""Given this query: "{query}"

Generate 2 alternative ways to phrase this same question that might appear in documents.
Consider different vocabulary, document structures, and professional contexts.
Focus on how this information might be presented in resumes, documents, or professional profiles.

Return only the alternative phrasings, one per line, without numbers or bullets."""
            
            # Use invoke instead of agenerate for more reliable results
            response = await self.llm.ainvoke(expansion_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            alternatives = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
            
            # Limit to 2 alternatives for performance
            alternatives = alternatives[:2]
            
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
    
    def analyze_query_term_importance(self, query: str, vectorstore) -> Dict[str, float]:
        """Content-agnostic analysis of query term importance based on corpus statistics"""
        # Extract meaningful terms from query
        terms = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        
        # Remove very common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'cannot'}
        meaningful_terms = [term for term in terms if term not in stop_words and len(term) > 2]
        
        term_importance = {}
        
        # Use simple heuristics for term importance (content-agnostic)
        for term in meaningful_terms:
            importance = 1.0
            
            # 1. Length-based importance (longer terms are often more specific)
            if len(term) >= 6:
                importance *= 1.5
            elif len(term) >= 4:
                importance *= 1.2
            
            # 2. Capitalization in original query suggests proper nouns/specificity
            if term.title() in query or term.upper() in query:
                importance *= 1.4
            
            # 3. Position-based importance (terms at start/end often more important)
            term_positions = [i for i, t in enumerate(terms) if t == term]
            if any(pos <= 1 or pos >= len(terms) - 2 for pos in term_positions):
                importance *= 1.1
            
            # 4. Frequency in query (repeated terms are important)
            term_frequency = terms.count(term)
            if term_frequency > 1:
                importance *= (1.0 + (term_frequency - 1) * 0.3)
            
            term_importance[term] = importance
        
        # Normalize importance scores
        if term_importance:
            max_importance = max(term_importance.values())
            for term in term_importance:
                term_importance[term] /= max_importance
        
        logger.info(f"[TERM_IMPORTANCE] Query: '{query}' -> Importance: {term_importance}")
        return term_importance
    
    def rerank_by_term_importance(self, query: str, retrieved_docs: List[Document], vectorstore) -> List[Document]:
        """Content-agnostic re-ranking based on term importance and document term density"""
        term_importance = self.analyze_query_term_importance(query, vectorstore)
        
        if not term_importance:
            return retrieved_docs
        
        logger.info(f"[IMPORTANCE_RERANKING] Re-ranking {len(retrieved_docs)} documents based on term importance")
        
        scored_docs = []
        for doc in retrieved_docs:
            content_lower = doc.page_content.lower()
            source = doc.metadata.get('file_name', '')
            source_lower = source.lower()
            
            # Calculate importance-weighted term score
            importance_score = 0.0
            term_matches = {}
            
            for term, importance in term_importance.items():
                # Count exact word matches (word boundaries)
                exact_matches = len(re.findall(rf'\b{re.escape(term)}\b', content_lower))
                
                if exact_matches > 0:
                    # Score = (importance * matches) with diminishing returns
                    term_score = importance * (1.0 + np.log(exact_matches))
                    importance_score += term_score
                    term_matches[term] = exact_matches
            
            # Context inference boost for person-specific queries
            context_boost = 0.0
            if self.classify_query_type(query) == 'relationship':
                # For queries like "who has X experience", apply context inference
                query_terms = [term.lower() for term in re.findall(r'\b[a-zA-Z]+\b', query)]
                
                # Check if query mentions a person and a technology/skill
                person_indicators = ['who', 'does', 'has']
                has_person_query = any(indicator in query_terms for indicator in person_indicators)
                
                if has_person_query and importance_score > 0:
                    # Boost documents where the person's name is in the filename
                    # This helps with cases where technology is mentioned in someone's resume
                    # but not explicitly connected to their personal experience
                    
                    # Check if this document is clearly from someone's resume/profile
                    is_personal_document = any(doc_type in source_lower for doc_type in ['resume', 'cv', 'profile'])
                    
                    if is_personal_document:
                        context_boost = importance_score * 0.5  # 50% boost for context inference
                        
                        # Additional boost for documents that mention specific technologies/skills
                        # (not just general terms like "experience")
                        specific_terms = [term for term, importance in term_importance.items() 
                                        if term not in ['who', 'has', 'does', 'experience', 'skilled', 'know', 'knows']]
                        
                        if specific_terms and any(term in term_matches for term in specific_terms):
                            # Extra boost for documents that mention specific technologies/skills
                            tech_boost = importance_score * 0.3  # Additional 30% for specific tech mentions
                            context_boost += tech_boost
                            logger.info(f"[CONTEXT_INFERENCE] Applied tech-specific boost {tech_boost:.2f} to {source}")
                        
                        logger.info(f"[CONTEXT_INFERENCE] Applied context boost {context_boost:.2f} to {source}")
            
            # Get original relevance score
            original_score = doc.metadata.get('relevance_score', 1.0)
            
            # Combine scores: boost documents with high-importance terms
            total_boost = importance_score + context_boost
            if total_boost > 0:
                # Boost proportional to total score
                final_score = original_score * (1.0 + total_boost * 0.3)
                boost_info = f" (importance_boost: {importance_score:.2f}"
                if context_boost > 0:
                    boost_info += f", context_boost: {context_boost:.2f}"
                boost_info += f", matches: {term_matches})"
            else:
                # Small penalty for documents without important terms
                final_score = original_score * 0.9
                boost_info = " (no_important_terms)"
            
            doc.metadata['final_relevance_score'] = final_score
            doc.metadata['importance_score'] = importance_score
            doc.metadata['context_boost'] = context_boost
            scored_docs.append((doc, final_score, boost_info))
        
        # Sort by final relevance score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Log the re-ranking results for debugging
        logger.info(f"[IMPORTANCE_RERANKING] Re-ranking results:")
        for i, (doc, score, boost_info) in enumerate(scored_docs[:5]):  # Log top 5
            source = doc.metadata.get('file_name', 'Unknown')
            logger.info(f"  {i+1}. {source} (Score: {score:.4f}){boost_info}")
        
        return [doc for doc, _, _ in scored_docs]
    
    async def multi_vector_search(self, query: str, vectorstore, k: int = 5) -> List[Document]:
        """Optimized search using multiple vector approaches with reduced API calls"""
        # Fast path for development mode
        if DEVELOPMENT_MODE:
            # Just do a single search in development mode for speed
            results = vectorstore.similarity_search(query, k=k)
            # Still apply content-agnostic re-ranking in development mode
            return self.rerank_by_term_importance(query, results, vectorstore)
        
        search_queries = [query]  # Start with original query
        
        # 2. Entity-focused search (combine all entities into one query)
        entities = self.extract_entities(query)
        if entities:
            entity_query = " ".join(entities[:3])  # Limit to top 3 entities for performance
            search_queries.append(entity_query)
        
        # 3. Concept-focused search (combine key concepts)
        concepts = self.extract_concepts(query)
        if concepts:
            concept_query = " ".join(concepts[:3])  # Limit to top 3 concepts for performance
            search_queries.append(concept_query)
        
        # 4. Add ONE semantic expansion instead of multiple
        if len(search_queries) <= 2:  # Only if we don't have many search terms already
            expanded_queries = await self.expand_query_semantically(query)
            if len(expanded_queries) > 1:
                search_queries.append(expanded_queries[1])  # Add only the first alternative
        
        # Batch execute all searches
        all_results = []
        try:
            # Execute searches in parallel using asyncio.gather for better performance
            search_tasks = []
            for search_query in search_queries:
                # Use smaller k per query to reduce total API calls
                per_query_k = max(1, k // len(search_queries))
                search_tasks.append(self._async_search(vectorstore, search_query, per_query_k))
            
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Collect results
            for result in search_results:
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"[ENHANCED_RETRIEVER] Search failed: {result}")
                    
        except Exception as e:
            logger.warning(f"[ENHANCED_RETRIEVER] Batch search failed: {e}")
            # Fallback to single search
            all_results = vectorstore.similarity_search(query, k=k)
        
        # Fast deduplication using set comprehension
        seen = set()
        unique_results = []
        for doc in all_results:
            doc_id = doc.page_content[:50]  # Use first 50 chars for faster comparison
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
                if len(unique_results) >= k:  # Early termination when we have enough results
                    break
        
        # Apply content-agnostic term importance re-ranking to final results
        reranked_results = self.rerank_by_term_importance(query, unique_results[:k], vectorstore)
        
        return reranked_results
    
    async def _async_search(self, vectorstore, query: str, k: int) -> List[Document]:
        """Async wrapper for vectorstore search"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, vectorstore.similarity_search, query, k)
        except Exception as e:
            logger.warning(f"[ENHANCED_RETRIEVER] Async search failed for query '{query}': {e}")
            return []
    
    async def hierarchical_search(self, query: str, vectorstore, k: int = 12) -> List[Document]:
        """Optimized hierarchical search with batch processing and reduced API calls"""
        # Fast path for development mode
        if DEVELOPMENT_MODE:
            # Skip complex hierarchical search in development mode
            return vectorstore.similarity_search(query, k=k)
        
        # Step 1: Smart entity/concept search (combine entities and concepts for fewer API calls)
        entities = self.extract_entities(query)
        concepts = self.extract_concepts(query)
        
        search_terms = []
        
        # Combine top entities into search terms (max 2 to limit API calls)
        if entities:
            search_terms.extend(entities[:2])
        
        # Add key concepts if we don't have enough entities
        if len(search_terms) < 2 and concepts:
            remaining_slots = 2 - len(search_terms)
            search_terms.extend(concepts[:remaining_slots])
        
        # If still no search terms, fall back to full query
        if not search_terms:
            search_terms = [query]
        
        # Execute searches in parallel
        broad_results = []
        try:
            search_tasks = []
            for term in search_terms:
                # Use smaller k per term to balance breadth vs API cost
                per_term_k = max(2, k // len(search_terms))
                search_tasks.append(self._async_search(vectorstore, term, per_term_k))
            
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            for result in search_results:
                if isinstance(result, list):
                    broad_results.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"[ENHANCED_RETRIEVER] Hierarchical search failed: {result}")
                    
        except Exception as e:
            logger.warning(f"[ENHANCED_RETRIEVER] Hierarchical search batch failed: {e}")
            # Fallback to simple search
            return vectorstore.similarity_search(query, k=k)
        
        # Step 2: Fast relevance filtering (batch similarity calculation)
        if not broad_results:
            return []
        
        threshold = self.get_adaptive_similarity_threshold(query)
        
        # Batch similarity calculation for better performance
        documents_text = [doc.page_content for doc in broad_results]
        
        try:
            # Batch embed all documents and the query
            query_texts = [query] + documents_text
            embeddings = await self.embedding_function.embed_queries_batch(query_texts)
            
            query_embedding = embeddings[0]
            doc_embeddings = embeddings[1:]
            
            # Calculate similarities in batch
            focused_results = []
            for i, (doc, doc_embedding) in enumerate(zip(broad_results, doc_embeddings)):
                try:
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    
                    if similarity > threshold:
                        doc.metadata['relevance_score'] = float(similarity)
                        focused_results.append(doc)
                        
                except Exception as e:
                    # Skip problematic documents
                    logger.warning(f"[ENHANCED_RETRIEVER] Similarity calculation failed: {e}")
                    continue
            
        except Exception as e:
            logger.warning(f"[ENHANCED_RETRIEVER] Batch similarity calculation failed: {e}")
            # Fallback to individual calculations (but limit to reduce cost)
            focused_results = []
            for doc in broad_results[:k*2]:  # Limit to avoid excessive API calls
                try:
                    relevance_score = self.calculate_semantic_similarity(query, doc.page_content)
                    if relevance_score > threshold:
                        doc.metadata['relevance_score'] = relevance_score
                        focused_results.append(doc)
                except Exception:
                    continue
        
        # Sort by relevance and deduplicate
        focused_results.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        
        # Fast deduplication
        seen = set()
        unique_results = []
        for doc in focused_results:
            doc_id = doc.page_content[:50]  # Shorter ID for faster comparison
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
                if len(unique_results) >= k:  # Early termination
                    break
        
        # Apply content-agnostic term importance re-ranking to final results
        reranked_results = self.rerank_by_term_importance(query, unique_results[:k], vectorstore)
        
        return reranked_results


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
