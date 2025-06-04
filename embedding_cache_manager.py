"""
Embedding Cache Manager

This module provides persistent caching for embeddings to significantly
reduce vector search time.
"""
import os
import numpy as np
import hashlib
import pickle
import time
from pathlib import Path

class EmbeddingCacheManager:
    def __init__(self, cache_dir="cache/embeddings", max_cache_size=1000):
        """
        Initialize the embedding cache manager.
        
        Args:
            cache_dir: Directory to store cached embeddings
            max_cache_size: Maximum number of embeddings to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.max_cache_size = max_cache_size
        self.memory_cache = {}
        self.access_times = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing cache metadata if available
        self.metadata_file = self.cache_dir / "metadata.pkl"
        self.metadata = self._load_metadata()
        
        print(f"Initialized embedding cache manager in {cache_dir}")
        print(f"Cache contains {len(self.metadata)} embeddings")
    
    def _load_metadata(self):
        """Load cache metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk"""
        try:
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"Error saving cache metadata: {e}")
    
    def _generate_key(self, text, model_name):
        """Generate a unique key for the text and model"""
        # Create a unique hash for the text and model
        text_bytes = text.encode('utf-8') if isinstance(text, str) else str(text).encode('utf-8')
        return hashlib.md5(f"{text_bytes}_{model_name}".encode()).hexdigest()
    
    def _get_cache_path(self, key):
        """Get the file path for a cached embedding"""
        return self.cache_dir / f"{key}.npy"
    
    def get(self, text, model_name):
        """
        Get an embedding from the cache.
        
        Args:
            text: Text to get embedding for
            model_name: Name of the embedding model
            
        Returns:
            Embedding if in cache, None otherwise
        """
        key = self._generate_key(text, model_name)
        
        # Check memory cache first
        if key in self.memory_cache:
            self.access_times[key] = time.time()
            return self.memory_cache[key]
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                embedding = np.load(cache_path)
                
                # Add to memory cache
                self._add_to_memory_cache(key, embedding)
                
                return embedding
            except Exception as e:
                print(f"Error loading cached embedding: {e}")
        
        return None
    
    def _add_to_memory_cache(self, key, embedding):
        """Add an embedding to the memory cache, evicting if necessary"""
        # Evict least recently used if cache is full
        if len(self.memory_cache) >= self.max_cache_size:
            # Get least recently used key
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            # Remove from memory cache and access times
            del self.memory_cache[lru_key]
            del self.access_times[lru_key]
        
        # Add to memory cache
        self.memory_cache[key] = embedding
        self.access_times[key] = time.time()
    
    def put(self, text, embedding, model_name):
        """
        Store an embedding in the cache.
        
        Args:
            text: Text the embedding is for
            embedding: The embedding vector
            model_name: Name of the embedding model
        """
        key = self._generate_key(text, model_name)
        cache_path = self._get_cache_path(key)
        
        try:
            # Save to disk
            np.save(cache_path, embedding)
            
            # Add to memory cache
            self._add_to_memory_cache(key, embedding)
            
            # Update metadata
            self.metadata[key] = {
                "model": model_name,
                "created": time.time(),
                "size": embedding.shape
            }
            
            # Save metadata periodically (every 10 additions)
            if len(self.metadata) % 10 == 0:
                self._save_metadata()
                
            return True
        except Exception as e:
            print(f"Error caching embedding: {e}")
            return False
    
    def clear(self):
        """Clear the entire cache"""
        # Clear memory cache
        self.memory_cache = {}
        self.access_times = {}
        
        # Clear disk cache
        for file in self.cache_dir.glob("*.npy"):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error removing cache file {file}: {e}")
        
        # Clear metadata
        self.metadata = {}
        if self.metadata_file.exists():
            try:
                os.remove(self.metadata_file)
            except Exception as e:
                print(f"Error removing metadata file: {e}")
    
    def get_stats(self):
        """Get cache statistics"""
        return {
            "disk_cache_size": len(self.metadata),
            "memory_cache_size": len(self.memory_cache),
            "cache_dir": str(self.cache_dir)
        }
