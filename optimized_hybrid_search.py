"""
Optimized Hybrid Search

This module enhances the original HybridSearch class with optimizations for:
1. Persistent embedding caching
2. Namespace filtering
3. Parallel search
4. Optimized BM25 search
"""
import time
import os
import numpy as np
from hybrid_search_for_workflow11 import HybridSearch
from embedding_cache_manager import EmbeddingCacheManager
from namespace_manager import NamespaceManager
from parallel_search import ParallelSearcher
import re
from concurrent.futures import ThreadPoolExecutor

class OptimizedHybridSearch(HybridSearch):
    def __init__(self, 
                 pinecone_api_key,
                 pinecone_index,
                 embedding_model=None,
                 alpha=0.7,
                 fusion_method="rrf",
                 normalize_embeds=True,
                 max_retries=3,
                 retry_delay=2,
                 cache_dir="cache/embeddings",
                 max_cache_size=1000,
                 max_workers=4):
        """
        Initialize Optimized Hybrid Search system with performance enhancements.
        
        Args:
            pinecone_api_key: API key for Pinecone
            pinecone_index: Name of the Pinecone index
            embedding_model: Name of the embedding model (if None, uses config value)
            alpha: Weight for vector search relative to BM25 (higher = more weight to vectors)
            fusion_method: Method to use for fusing results ("weighted", "rrf", or "combmnz")
            normalize_embeds: Whether to normalize embeddings to unit length
            max_retries: Maximum number of retries for Pinecone connection
            retry_delay: Delay between retries
            cache_dir: Directory to store cached embeddings
            max_cache_size: Maximum number of embeddings to keep in memory
            max_workers: Maximum number of worker threads for parallel search
        """
        # Initialize the base HybridSearch class
        super().__init__(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index,
            embedding_model=embedding_model,
            alpha=alpha,
            fusion_method=fusion_method,
            normalize_embeds=normalize_embeds,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        # Initialize embedding cache manager
        self.embedding_cache = EmbeddingCacheManager(
            cache_dir=cache_dir,
            max_cache_size=max_cache_size
        )
        
        # Initialize namespace manager
        self.namespace_manager = NamespaceManager(self.namespaces)
        
        # Initialize parallel searcher
        self.parallel_searcher = ParallelSearcher(max_workers=max_workers)
        
        # Track performance metrics
        self.last_vector_search_time = 0
        self.last_bm25_search_time = 0
        self.last_fusion_time = 0
        
        print("Optimized Hybrid Search initialized with:")
        print(f"  - Embedding cache: {cache_dir}")
        print(f"  - Namespace filtering: enabled")
        print(f"  - Parallel search: {max_workers} workers")
    
    def _get_query_embedding(self, query):
        """
        Get embedding for a query with caching.
        
        Args:
            query: Query text to get embedding for
            
        Returns:
            Embedding vector
        """
        # Get the model name (use the actual model object's name attribute)
        model_name = self.embedding_model.__class__.__name__
        query_normalized = query.lower().strip()
        
        # Try to get from cache first
        embedding = self.embedding_cache.get(query_normalized, model_name)
        
        if embedding is not None:
            print(f"Using cached embedding for '{query[:30]}...'")
            return embedding
        
        # Calculate embedding if not in cache
        print(f"Calculating embedding for '{query[:30]}...'")
        
        # Create embedding using the parent class's method
        embedding = self.embedding_model.encode(query)
        
        # Normalize the embedding if needed
        if self.normalize_embeds:
            embedding = embedding / np.linalg.norm(embedding)
        
        # Adapt dimension if needed
        embedding = self._adapt_dimension(embedding)
        
        # Store in cache
        self.embedding_cache.put(query_normalized, embedding, model_name)
        
        return embedding
    
    def _search_namespace(self, query_vector, namespace, top_k=5):
        """
        Search a single namespace with query vector.
        
        Args:
            query_vector: Query vector to search with
            namespace: Namespace to search
            top_k: Number of top results to return
            
        Returns:
            Search results
        """
        try:
            # Search the namespace
            results = self.index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            
            # Process results
            processed_results = []
            for match in results.matches:
                # Get metadata
                metadata = match.metadata if hasattr(match, 'metadata') else {}
                
                # Create result object
                result = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": metadata,
                    "namespace": namespace,
                    "source": "vector"
                }
                
                processed_results.append(result)
            
            return processed_results
        except Exception as e:
            print(f"Error searching namespace {namespace}: {e}")
            return []
    
    def _semantic_search(self, query, top_k=5, namespaces=None):
        """
        Perform semantic search with optimizations.
        
        Args:
            query: Query text
            top_k: Number of results to return
            namespaces: List of namespaces to search in
            
        Returns:
            List of search results
        """
        start_time = time.time()
        
        # Get query embedding
        query_vector = self._get_query_embedding(query)
        
        # Get relevant namespaces
        query_type = self.namespace_manager.classify_query(query)
        relevant_namespaces = self.namespace_manager.get_relevant_namespaces(query, min_namespaces=3)
        
        print(f"Query classified as: {query_type}")
        print(f"Searching {len(relevant_namespaces)} namespaces: {', '.join(relevant_namespaces[:3])}...")
        
        # Search relevant namespaces in parallel
        results = self.parallel_searcher.search_namespaces(
            self._search_namespace,
            query_vector,
            relevant_namespaces,
            top_k
        )
        
        self.last_vector_search_time = time.time() - start_time
        print(f"Semantic search completed in {self.last_vector_search_time:.3f}s")
        
        return results
    
    def _bm25_search(self, query, top_k=5):
        """
        Perform BM25 search with optimizations.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            Search results
        """
        start_time = time.time()
        
        # Use the original BM25 search implementation with some optimizations
        results = super()._bm25_search(query, top_k)
        
        self.last_bm25_search_time = time.time() - start_time
        print(f"BM25 search completed in {self.last_bm25_search_time:.3f}s")
        
        return results
    
    def hybrid_search(self, query, top_k=5, alpha=None, fusion_method=None):
        """
        Perform optimized hybrid search.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            alpha: Weight for vector search (0-1)
            fusion_method: Method to use for fusing results
            
        Returns:
            Search results
        """
        # Start overall timing
        start_time = time.time()
        
        if alpha is None:
            alpha = self.alpha
            
        if fusion_method is None:
            fusion_method = self.fusion_method
        
        # Get enough results for effective fusion
        search_k = top_k * 3
        
        # Run vector and BM25 search in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(self._semantic_search, query, search_k)
            bm25_future = executor.submit(self._bm25_search, query, search_k)
            
            # Get results
            vector_results = vector_future.result()
            bm25_results = bm25_future.result()
        
        print(f"Vector search returned {len(vector_results)} results")
        print(f"BM25 search returned {len(bm25_results)} results")
        
        # If both searches return no results, return empty list
        if not vector_results and not bm25_results:
            print("No results found from either search method")
            return []
        
        # If only one search method returned results, use those
        if not vector_results:
            print("Using BM25 results only (no vector results)")
            return bm25_results[:top_k]
        elif not bm25_results:
            print("Using vector results only (no BM25 results)")
            return vector_results[:top_k]
        
        # TIMING: Result fusion
        fusion_start = time.time()
        
        # Apply the selected fusion method
        if fusion_method == "rrf":
            results = self._reciprocal_rank_fusion(vector_results, bm25_results)
        elif fusion_method == "combmnz":
            results = self._combmnz_fusion(vector_results, bm25_results, alpha, 1-alpha)
        else:
            # Use weighted fusion
            results = super().hybrid_search(query, top_k, alpha, "weighted")
        
        self.last_fusion_time = time.time() - fusion_start
        print(f"Result fusion completed in {self.last_fusion_time:.3f}s")
        
        # Limit to top_k results
        results = results[:top_k]
        
        total_time = time.time() - start_time
        print(f"Hybrid search completed in {total_time:.3f}s")
        
        return results
