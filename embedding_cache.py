#!/usr/bin/env python3
"""
Embedding Cache for the Hybrid Search System
This module provides caching functionality for embeddings to improve performance.
"""
import hashlib
import numpy as np
from functools import lru_cache

# Cache for embeddings
embedding_cache = {}
MAX_EMBEDDING_CACHE_SIZE = 500

def get_embedding_cache_key(text):
    """Generate a cache key for text embeddings."""
    return hashlib.md5(text.encode()).hexdigest()

def add_to_embedding_cache(key, value):
    """Add an embedding to the cache with management of cache size."""
    embedding_cache[key] = value
    # Manage cache size
    if len(embedding_cache) > MAX_EMBEDDING_CACHE_SIZE:
        # Remove oldest item (first key)
        embedding_cache.pop(next(iter(embedding_cache)))

# Create a cached version of the embedding function
@lru_cache(maxsize=500)
def cached_encode(text, model):
    """Cached version of the encode function for embeddings."""
    return model.encode(text)

def apply_embedding_caching():
    """
    Apply embedding caching by patching the encode method of the embedding model.
    """
    from hybrid_search_for_workflow11 import HybridSearch
    from sentence_transformers import SentenceTransformer
    
    # Store the original encode method of SentenceTransformer
    original_encode = SentenceTransformer.encode
    
    # Define the cached version
    def cached_encode(self, texts, *args, **kwargs):
        """Cached version of the encode method."""
        # Handle both single text and batch texts
        if isinstance(texts, str):
            cache_key = get_embedding_cache_key(texts)
            
            # Check cache
            if cache_key in embedding_cache:
                print("Using cached embedding")
                return embedding_cache[cache_key]
            
            # Generate embedding using original method
            embedding = original_encode(self, texts, *args, **kwargs)
            
            # Cache the embedding
            add_to_embedding_cache(cache_key, embedding)
            
            return embedding
        else:
            # For batches, process each text individually for better caching
            results = []
            for text in texts:
                cache_key = get_embedding_cache_key(text)
                
                # Check cache
                if cache_key in embedding_cache:
                    print("Using cached embedding")
                    results.append(embedding_cache[cache_key])
                else:
                    # Generate embedding using original method
                    embedding = original_encode(self, text, *args, **kwargs)
                    
                    # Cache the embedding
                    add_to_embedding_cache(cache_key, embedding)
                    
                    results.append(embedding)
            
            # Convert list of embeddings to numpy array
            return np.array(results)
    
    # Replace the original method with our cached version
    SentenceTransformer.encode = cached_encode
    
    print("Embedding caching applied")
