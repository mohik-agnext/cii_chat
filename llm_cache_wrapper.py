#!/usr/bin/env python3
"""
LLM Cache Wrapper for the Hybrid Search System
This module provides caching functionality for LLM responses to improve performance.
"""
import hashlib
import time
import json
from functools import lru_cache

# Import the original function to wrap
from hybrid_search_server import generate_llm_response as original_generate_llm_response

# Cache for LLM responses
llm_cache = {}
MAX_CACHE_SIZE = 100

def get_cache_key(query, search_results, session_id):
    """Generate a cache key from query and search results."""
    # Create a deterministic string representation of search results
    # We only use the IDs and scores to avoid changes in metadata affecting the cache
    results_str = ""
    for result in search_results:
        results_str += f"{result.get('id', '')}:{result.get('score', 0):.4f};"
    
    # Combine query and results into a single key
    key = f"{query.lower().strip()}_{results_str}"
    return hashlib.md5(key.encode()).hexdigest()

def add_to_cache(key, value):
    """Add a value to the cache with management of cache size."""
    llm_cache[key] = value
    # Manage cache size
    if len(llm_cache) > MAX_CACHE_SIZE:
        # Remove oldest item (first key)
        llm_cache.pop(next(iter(llm_cache)))

def cached_generate_llm_response(query, search_results, session_id, model=None, api_key=None, stream=False):
    """
    Cached version of generate_llm_response function.
    Checks cache before generating a new response.
    """
    # Don't use cache for streaming responses
    if stream:
        return original_generate_llm_response(query, search_results, session_id, model, api_key, stream)
    
    # Generate cache key
    cache_key = get_cache_key(query, search_results, session_id)
    
    # Check cache
    if cache_key in llm_cache:
        print("Using cached LLM response")
        return llm_cache[cache_key]
    
    # Measure response time
    start_time = time.time()
    
    # Generate response using original function
    response = original_generate_llm_response(query, search_results, session_id, model, api_key, stream)
    
    # Calculate time taken
    time_taken = time.time() - start_time
    print(f"LLM response generated in {time_taken:.2f}s")
    
    # Cache the response
    add_to_cache(cache_key, response)
    
    return response

# Function to patch the original function with our cached version
def apply_llm_caching():
    """
    Apply LLM caching by replacing the original function with our cached version.
    """
    import hybrid_search_server
    import sys
    
    # Replace the original function with our cached version
    sys.modules['hybrid_search_server'].generate_llm_response = cached_generate_llm_response
    
    print("LLM response caching applied")
