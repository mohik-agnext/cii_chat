"""
Parallel Search

This module provides parallel search capabilities to accelerate vector search
by searching multiple namespaces concurrently.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelSearcher:
    def __init__(self, max_workers=4):
        """
        Initialize the parallel searcher.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        print(f"Initialized parallel searcher with {max_workers} workers")
    
    def search_namespaces(self, search_func, query_vector, namespaces, top_k):
        """
        Search multiple namespaces in parallel.
        
        Args:
            search_func: Function to call for searching a namespace
            query_vector: Query vector to search with
            namespaces: List of namespaces to search
            top_k: Number of top results to return per namespace
            
        Returns:
            Combined search results
        """
        start_time = time.time()
        
        # Limit workers to number of namespaces
        workers = min(self.max_workers, len(namespaces))
        
        all_results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit search tasks
            future_to_namespace = {
                executor.submit(search_func, query_vector, namespace, top_k): namespace
                for namespace in namespaces
            }
            
            # Process results as they complete
            for future in as_completed(future_to_namespace):
                namespace = future_to_namespace[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"Namespace {namespace}: found {len(results)} results")
                except Exception as e:
                    errors.append((namespace, str(e)))
                    print(f"Error searching namespace {namespace}: {e}")
        
        # Report errors
        if errors:
            print(f"Encountered {len(errors)} errors during parallel search")
            for namespace, error in errors:
                print(f"  - {namespace}: {error}")
        
        # Sort results by score
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Truncate to top_k
        results = all_results[:top_k]
        
        search_time = time.time() - start_time
        print(f"Parallel search completed in {search_time:.3f}s, found {len(results)} results from {len(namespaces)} namespaces")
        
        return results
    
    def hybrid_search_parallel(self, vector_search_func, bm25_search_func, query, namespaces, top_k, alpha=0.5):
        """
        Perform hybrid search with vector and BM25 components running in parallel.
        
        Args:
            vector_search_func: Function to call for vector search
            bm25_search_func: Function to call for BM25 search
            query: Search query
            namespaces: List of namespaces to search
            top_k: Number of top results to return
            alpha: Weight for vector search (0-1)
            
        Returns:
            Combined search results
        """
        start_time = time.time()
        
        vector_results = []
        bm25_results = []
        
        # Run vector and BM25 search in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(vector_search_func, query, namespaces, top_k*2)
            bm25_future = executor.submit(bm25_search_func, query, top_k*2)
            
            # Get results
            vector_results = vector_future.result()
            bm25_results = bm25_future.result()
        
        print(f"Vector search found {len(vector_results)} results")
        print(f"BM25 search found {len(bm25_results)} results")
        
        # Combine results (simplified version - actual fusion would be done by the main system)
        combined_results = vector_results + bm25_results
        
        search_time = time.time() - start_time
        print(f"Parallel hybrid search completed in {search_time:.3f}s")
        
        return combined_results
