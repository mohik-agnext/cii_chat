#!/usr/bin/env python3
import json
import numpy as np
from typing import List, Dict, Any
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time
from functools import wraps
import config

# Main function that will be called by n8n
def execute(params):
    try:
        # Extract input parameters
        query = params.get('query', '')
        if not query:
            # If there's a body.message, use that (for compatibility with existing workflow)
            if isinstance(params.get('body'), dict):
                query = params.get('body', {}).get('message', '')
        
        # Get Pinecone credentials from n8n credentials
        pinecone_api_key = params.get('pineconeApiKey', '')
        pinecone_index = params.get('pineconeIndex', 'cursor2') # Default to the index in your workflow
        
        # Get search parameters (with defaults)
        top_k = int(params.get('top_k', 4))  # Default to 4 as in your workflow
        alpha = float(params.get('alpha', 0.5))
        fusion_method = params.get('fusion_method', 'rrf')  # Changed default to RRF based on testing
        
        # Create hybrid searcher
        searcher = HybridSearch(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index,
            alpha=alpha,
            fusion_method=fusion_method
        )
        
        # Perform search
        results = searcher.hybrid_search(query, top_k=top_k, alpha=alpha, fusion_method=fusion_method)
        
        # Format results to be compatible with your existing workflow
        documents = []
        for result in results:
            # Extract text field from metadata (this is what your workflow expects)
            text = result.get('metadata', {}).get('text', '')
            if text:
                # Include all relevant metadata
                source_info = result.get('source', 'hybrid')
                source_text = ""
                
                # Add source information based on where the result came from
                if source_info == "bm25":
                    source_text = "BM25 Lexical Search"
                elif source_info == "vector":
                    source_text = "Vector Semantic Search"
                elif "hybrid" in source_info:
                    source_text = f"Hybrid Search ({source_info})"
                
                # Include document ID and namespace for reference
                doc_id = result.get('id', '')
                namespace = result.get('namespace', '')
                
                documents.append({
                    'pageContent': text,
                    'metadata': {
                        'id': doc_id,
                        'score': result.get('score', 0),
                        'source': source_info,
                        'source_description': source_text,
                        'namespace': namespace,
                        # Include raw BM25 score if available
                        'raw_bm25_score': result.get('raw_bm25_score', None)
                    }
                })
        
        # Return in a format compatible with AI Agent node
        return {
            'query': query,
            'documents': documents,
            'hybrid_search': True,
            'fusion_method': fusion_method,
            'alpha': alpha,
            'sources_included': True  # Flag to indicate sources are included
        }
        
    except Exception as e:
        return {'error': str(e), 'message': 'Error in hybrid search execution'}

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep_time = (backoff_in_seconds * 2 ** x)
                    print(f"Retrying {func.__name__} in {sleep_time} seconds... (Attempt {x + 1}/{retries})")
                    time.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator

class HybridSearch:
    def __init__(self, 
                 pinecone_api_key,
                 pinecone_index,
                 embedding_model=None,
                 alpha=0.7,
                 fusion_method="rrf",
                 normalize_embeds=True,
                 max_retries=3,
                 retry_delay=2):
        """
        Initialize Hybrid Search system that combines BM25 lexical search with vector search.
        
        Args:
            pinecone_api_key: API key for Pinecone
            pinecone_index: Name of the Pinecone index
            embedding_model: Name of the embedding model (if None, uses config value)
            alpha: Weight for vector search relative to BM25 (higher = more weight to vectors)
            fusion_method: Method to use for fusing results ("weighted", "rrf", or "combmnz")
            normalize_embeds: Whether to normalize embeddings to unit length
            max_retries: Maximum number of retries for Pinecone connection
            retry_delay: Delay between retries
        """
        self.alpha = alpha
        self.fusion_method = fusion_method
        self.normalize_embeds = normalize_embeds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.embedding_dimension = config.PINECONE_DIMENSION  # Set dimension from config
        
        # Initialize BM25 variables
        self.bm25_index = None
        self.bm25_documents = []
        self.doc_ids = []
        
        # OPTIMIZATION: Add embedding cache
        self.query_embedding_cache = {}
        self.max_cache_size = 100  # Maximum number of cached embeddings
        
        print(f"\nInitializing Hybrid Search with index '{pinecone_index}'")
        
        # Initialize Pinecone with retries
        self._initialize_pinecone(pinecone_api_key, pinecone_index)
        
        # Use provided embedding model or fall back to config
        if embedding_model is None:
            embedding_model = config.EMBEDDING_MODEL
        
        # Initialize embedding model with timing
        print(f"Loading embedding model: {embedding_model}")
        model_load_start = time.time()
        self.embedding_model = SentenceTransformer(embedding_model)
        model_load_time = time.time() - model_load_start
        print(f"Embedding model loaded in {model_load_time:.2f} seconds")
        
        # Get the namespaces in the index with retries
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                stats = self.index.describe_index_stats()
                self.namespaces = list(stats.namespaces.keys())
                print(f"Successfully retrieved namespaces: {self.namespaces}")
                break
            except Exception as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    print(f"Attempt {retry_count} failed to get namespaces: {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to get namespaces after {self.max_retries} attempts: {e}")
                    self.namespaces = []
        
        # Benchmark embedding performance
        test_text = "Test embedding dimension and performance benchmark"
        encode_start = time.time()
        test_embedding = self.embedding_model.encode(test_text)
        encode_time = time.time() - encode_start
        
        self.embedding_dimension_actual = len(test_embedding)
        print(f"Embedding dimension: {self.embedding_dimension_actual}")
        print(f"Single encoding time: {encode_time*1000:.2f}ms")
        
        # Benchmark batch encoding (more realistic scenario)
        batch_texts = [
            "What are the eligibility criteria?",
            "How do I apply for this program?",
            "What documents are required?",
            "What are the fees for this service?",
            "When is the deadline for application?"
        ]
        batch_start = time.time()
        _ = self.embedding_model.encode(batch_texts)
        batch_time = time.time() - batch_start
        print(f"Batch encoding time (5 queries): {batch_time*1000:.2f}ms ({batch_time*200:.2f}ms/query)")
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer")
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK stopwords")
            nltk.download('stopwords')
            
        # Initialize BM25 after everything else is set up
        self._initialize_bm25()
    
    @retry_with_backoff(retries=3, backoff_in_seconds=1)
    def _initialize_pinecone(self, api_key: str, index_name: str):
        """Initialize Pinecone with retries."""
        print(f"Initializing Pinecone with index: {index_name} in environment: {config.PINECONE_ENVIRONMENT}")
        
        try:
            # Initialize Pinecone
            pc = Pinecone(api_key=api_key)
            
            # Connect to index
            self.index = pc.Index(name=index_name)
            
            # Get the namespaces in the index
            try:
                stats = self.index.describe_index_stats()
                self.namespaces = list(stats.namespaces.keys())
                print(f"Connected to Pinecone index with {len(self.namespaces)} namespaces")
                
                # Set index dimension
                self.index_dimension = config.PINECONE_DIMENSION
                print(f"Index dimension: {self.index_dimension}")
                
            except Exception as e:
                print(f"Error getting index stats: {e}")
                self.namespaces = []
                self.index_dimension = config.PINECONE_DIMENSION  # Use configured dimension
                print(f"Using configured dimension: {self.index_dimension}")
                
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            self.index_dimension = config.PINECONE_DIMENSION  # Use configured dimension
            print(f"Using configured dimension: {self.index_dimension}")
    
    def _get_namespaces(self, stats):
        """Get available namespaces in the index."""
        return list(stats.namespaces.keys())
    
    def _adapt_dimension(self, vector, target_dim=None):
        """
        Adapt vector dimension to match target dimension by padding or truncating.
        
        Args:
            vector: The vector to adapt
            target_dim: Target dimension (defaults to index dimension)
        
        Returns:
            Vector with adapted dimension
        """
        if target_dim is None:
            target_dim = self.index_dimension
        
        if target_dim is None:
            # If we don't know the index dimension, return as is
            return vector
        
        current_dim = len(vector)
        
        if current_dim == target_dim:
            # No adaptation needed
            return vector
        
        elif current_dim < target_dim:
            # OPTIMIZATION: More efficient padding for small embeddings
            # Print adaptation info only once
            if not hasattr(self, '_dimension_adapted_logged'):
                print(f"Adapting embedding dimension: {current_dim} → {target_dim} (padding)")
                self._dimension_adapted_logged = True
                
            # Create padding array once and reuse
            if not hasattr(self, '_padding_array') or len(self._padding_array) != (target_dim - current_dim):
                self._padding_array = np.zeros(target_dim - current_dim)
                
            # Use np.concatenate for efficient padding
            return np.concatenate([vector, self._padding_array])
        
        else:
            # OPTIMIZATION: More efficient truncation
            # Print adaptation info only once
            if not hasattr(self, '_dimension_adapted_logged'):
                print(f"Adapting embedding dimension: {current_dim} → {target_dim} (truncating)")
                self._dimension_adapted_logged = True
                
            # Simple truncation
            return vector[:target_dim]
    
    def verify_bm25_index(self):
        """Verify that the BM25 index is working correctly."""
        print("\nVerifying BM25 index...")
        
        if not self.bm25_index or not self.bm25_documents:
            print("BM25 index not initialized")
            return False
            
        print(f"BM25 index contains {len(self.bm25_documents)} documents")
        
        # Try a test search
        test_query = "test query for verification"
        try:
            # Tokenize query
            stop_words = set(stopwords.words('english'))
            query_tokens = word_tokenize(test_query.lower())
            query_tokens = [token for token in query_tokens if token.isalnum() and token not in stop_words]
            
            # Get scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            if len(scores) != len(self.bm25_documents):
                print(f"Error: Score count ({len(scores)}) doesn't match document count ({len(self.bm25_documents)})")
                return False
                
            print("BM25 index verification successful")
            return True
            
        except Exception as e:
            print(f"Error verifying BM25 index: {e}")
            return False
            
    def _initialize_bm25(self):
        """Initialize BM25 index with documents from Pinecone."""
        print("\nInitializing BM25 index...")
        
        try:
            # Get all documents from Pinecone
            self.bm25_documents = []
            self.doc_ids = []
            
            # Get documents from each namespace
            for namespace in self.namespaces:
                try:
                    # Get all vector IDs in this namespace
                    stats = self.index.describe_index_stats()
                    vector_count = stats.namespaces[namespace].vector_count
                    
                    print(f"Found {vector_count} vectors in namespace '{namespace}'")
                    
                    # Fetch vectors in batches of 1000
                    batch_size = 1000
                    for i in range(0, vector_count, batch_size):
                        try:
                            # Query for a batch of vectors
                            results = self.index.query(
                                vector=[0.0] * self.embedding_dimension,  # Dummy vector
                                top_k=min(batch_size, vector_count - i),
                                namespace=namespace,
                                include_metadata=True
                            )
                            
                            # Extract text and ids
                            batch_count = 0
                            for match in results.matches:
                                if 'text' in match.metadata:
                                    # Only add documents with sufficient content
                                    text = match.metadata['text'].strip()
                                    if len(text) > 10:  # Minimum text length to be indexed
                                        self.bm25_documents.append(text)
                                        self.doc_ids.append((match.id, namespace))
                                        batch_count += 1
                            
                            print(f"Added {batch_count} documents from batch {i//batch_size + 1}")
                                    
                        except Exception as e:
                            print(f"Error fetching batch from namespace {namespace}: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Error processing namespace {namespace}: {e}")
                    continue
            
            if not self.bm25_documents:
                print("Warning: No documents found for BM25 indexing")
                return
                
            print(f"Found {len(self.bm25_documents)} documents for BM25 indexing")
            
            # Tokenize documents for BM25
            try:
                # First try to get stopwords
                try:
                    stop_words = set(stopwords.words('english'))
                except Exception as e:
                    print(f"Warning: Failed to load stopwords, continuing without them: {e}")
                    stop_words = set()  # Empty set as fallback
                
                # Add common stopwords that might not be in NLTK's list
                additional_stopwords = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of'}
                stop_words.update(additional_stopwords)
                
                tokenized_docs = []
                skipped_docs = 0
                
                for doc in self.bm25_documents:
                    try:
                        # Try NLTK tokenization first
                        try:
                            # Convert to lowercase and tokenize
                            tokens = word_tokenize(doc.lower())
                        except Exception as e:
                            print(f"Warning: NLTK tokenization failed, using fallback method: {e}")
                            # Simple fallback tokenization
                            tokens = doc.lower().split()
                        
                        # Filter tokens - improved filtering logic
                        filtered_tokens = []
                        for token in tokens:
                            # Keep tokens that are:
                            # 1. At least 2 characters long
                            # 2. Alphanumeric or contain hyphens (for compound words)
                            # 3. Not in stopwords
                            # 4. Not just numbers
                            if (len(token) > 1 and 
                                (token.isalnum() or '-' in token) and 
                                token not in stop_words and
                                not token.isdigit()):
                                filtered_tokens.append(token)
                        
                        if len(filtered_tokens) >= 3:  # Only add if we have enough tokens
                            tokenized_docs.append(filtered_tokens)
                        else:
                            skipped_docs += 1
                        
                    except Exception as e:
                        print(f"Error tokenizing document: {e}")
                        skipped_docs += 1
                        continue
                
                print(f"Skipped {skipped_docs} documents due to insufficient tokens")
                
                # Create BM25 index
                if tokenized_docs:
                    print(f"Creating BM25 index with {len(tokenized_docs)} tokenized documents")
                    
                    # Check if we need to adjust the document list
                    if len(tokenized_docs) < len(self.bm25_documents):
                        print(f"Adjusting document list to match tokenized documents ({len(tokenized_docs)} vs {len(self.bm25_documents)})")
                        # Create new lists with only the documents that were successfully tokenized
                        new_docs = []
                        new_ids = []
                        
                        doc_index = 0
                        for i, doc in enumerate(self.bm25_documents):
                            try:
                                # Check if this document was tokenized successfully
                                tokens = word_tokenize(doc.lower())
                                filtered_tokens = [t for t in tokens if len(t) > 1 and (t.isalnum() or '-' in t) and t not in stop_words and not t.isdigit()]
                                
                                if len(filtered_tokens) >= 3:
                                    new_docs.append(doc)
                                    new_ids.append(self.doc_ids[i])
                                    doc_index += 1
                            except:
                                # Skip documents that can't be tokenized
                                continue
                                
                        # Update the document lists
                        self.bm25_documents = new_docs
                        self.doc_ids = new_ids
                        print(f"Document list adjusted to {len(self.bm25_documents)} documents")
                    
                    # Create the BM25 index with the tokenized documents
                    self.bm25_index = BM25Okapi(tokenized_docs)
                    print("BM25 index initialized successfully")
                    
                    # Verify the index
                    if self.verify_bm25_index():
                        print("BM25 index verification successful")
                    else:
                        print("Warning: BM25 index verification failed")
                else:
                    print("Warning: No valid tokenized documents to create BM25 index")
                    self.bm25_index = None
                    
            except Exception as e:
                print(f"Error during document tokenization: {e}")
                self.bm25_index = None
                
        except Exception as e:
            print(f"Error during BM25 initialization: {e}")
            self.bm25_index = None
            self.bm25_documents = []
            self.doc_ids = []
    
    @retry_with_backoff(retries=3, backoff_in_seconds=1)
    def _bm25_search(self, query: str, top_k: int = 5):
        """Perform lexical search using BM25 with retries."""
        print(f"\nPerforming BM25 search for query: {query}")
        
        if not self.bm25_index or not self.bm25_documents:
            print("Warning: BM25 index not initialized")
            return []
        
        try:
            # Tokenize and prepare query
            try:
                # Try to get stopwords
                try:
                    stop_words = set(stopwords.words('english'))
                    # Add common stopwords that might not be in NLTK's list
                    additional_stopwords = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of'}
                    stop_words.update(additional_stopwords)
                except Exception as e:
                    print(f"Warning: Failed to load stopwords for query, continuing without them: {e}")
                    stop_words = set()  # Empty set as fallback
                
                # Try NLTK tokenization first
                try:
                    query_tokens = word_tokenize(query.lower())
                except Exception as e:
                    print(f"Warning: NLTK query tokenization failed, using fallback method: {e}")
                    # Simple fallback tokenization
                    query_tokens = query.lower().split()
                
                # Filter tokens - improved filtering logic
                filtered_tokens = []
                for token in query_tokens:
                    # Keep tokens that are:
                    # 1. At least 2 characters long
                    # 2. Alphanumeric or contain hyphens (for compound words)
                    # 3. Not in stopwords
                    # 4. Not just numbers
                    if (len(token) > 1 and 
                        (token.isalnum() or '-' in token) and 
                        token not in stop_words and
                        not token.isdigit()):
                        filtered_tokens.append(token)
                
                query_tokens = filtered_tokens
            
            except Exception as e:
                print(f"Error tokenizing query, using simple approach: {e}")
                # Last resort tokenization
                query_tokens = [w.lower() for w in query.split() if len(w) > 1]
            
            if not query_tokens:
                print("Warning: No valid tokens in query after processing")
                return []
                
            print(f"Tokenized query: {query_tokens}")
            
            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k document indices
            top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
            
            if len(top_indices) == 0:
                print("No matching documents found for the query")
                return []
            
            # Filter out documents with zero or very low scores (threshold can be adjusted)
            min_score_threshold = 0.1
            filtered_indices = [idx for idx in top_indices if bm25_scores[idx] > min_score_threshold]
            
            if not filtered_indices:
                print(f"No documents with scores above threshold {min_score_threshold}")
                # Fall back to using all results if none pass the threshold
                filtered_indices = top_indices
                
            print(f"Top BM25 scores: {[bm25_scores[i] for i in filtered_indices]}")
            
            # Find the maximum BM25 score for better normalization
            max_bm25_score = max([bm25_scores[i] for i in filtered_indices]) if filtered_indices else 1.0
            # Use a minimum threshold to avoid division by very small numbers
            max_bm25_score = max(max_bm25_score, 1.0)
            
            # Retrieve original documents and their ids
            results = []
            used_ids = set()  # Track used IDs to prevent duplicates
            
            for idx in filtered_indices:
                if bm25_scores[idx] > 0:  # Only include if there's some similarity
                    doc_id, namespace = self.doc_ids[idx]
                    
                    # Skip if we've already included this document
                    if doc_id in used_ids:
                        continue
                    
                    used_ids.add(doc_id)
                    
                    # Find the corresponding vector in Pinecone
                    try:
                        vector_results = self.index.fetch(ids=[doc_id], namespace=namespace)
                        
                        # Check if the document exists in the fetch response
                        if doc_id in vector_results.vectors:
                            vector_data = vector_results.vectors[doc_id]
                            # Normalize BM25 score to 0-1 range using the max score
                            normalized_score = bm25_scores[idx] / max_bm25_score
                            
                            # Create a detailed result with both normalized and raw scores
                            result = {
                                "id": doc_id,
                                "score": float(normalized_score),
                                "metadata": vector_data.metadata,
                                "namespace": namespace,
                                "source": "bm25",
                                "raw_bm25_score": float(bm25_scores[idx]),
                                "search_method": "lexical",
                                "tokens_matched": len(query_tokens),
                                "query_tokens": query_tokens
                            }
                            results.append(result)
                            print(f"BM25 result: {doc_id}, score: {normalized_score:.4f}, raw: {bm25_scores[idx]:.4f}")
                    except Exception as e:
                        print(f"Error fetching document {doc_id} from namespace {namespace}: {e}")
                        continue
            
            print(f"BM25 search returned {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Error during BM25 search: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, vector_results: List[dict], bm25_results: List[dict], k: int = None):
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF(d) = sum(1/(k + r_i)) where r_i is the rank of d in the ith result list.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: Constant to mitigate impact of high rankings (default from config or 60)
            
        Returns:
            Combined and reranked results
        """
        # Use config RRF_K_VALUE if available
        if k is None:
            k = getattr(config, 'RRF_K_VALUE', 60)
            
        # Rest of method unchanged
        all_ids = set()
        for result in vector_results:
            all_ids.add(result["id"])
        for result in bm25_results:
            all_ids.add(result["id"])
        
        # Calculate RRF scores
        rrf_scores = {}
        
        # Add vector search results scores
        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
        # Add BM25 search results scores
        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Create final results list
        results = []
        for doc_id in sorted_ids:
            # Find original result to preserve metadata
            original_result = None
            
            # Check vector results first
            for result in vector_results:
                if result["id"] == doc_id:
                    original_result = result
                    original_result["source"] = "vector"
                    break
                    
            # If not found in vector results, check BM25 results
            if original_result is None:
                for result in bm25_results:
                    if result["id"] == doc_id:
                        original_result = result
                        original_result["source"] = "bm25"
                        break
            
            if original_result:
                # Add RRF score
                original_result["score"] = rrf_scores[doc_id]
                results.append(original_result)
                print(f"RRF fusion result: {doc_id}, score: {rrf_scores[doc_id]:.4f}, source: {original_result['source']}")
            
        return results
    
    def _combmnz_fusion(self, vector_results: List[dict], bm25_results: List[dict], 
                      vector_weight: float = 0.6, bm25_weight: float = 0.4):
        """
        Combine results using CombMNZ fusion.
        """
        doc_scores = {}
        
        # Normalize scores within each method
        vector_max = max((r["score"] for r in vector_results), default=1.0)
        bm25_max = max((r["score"] for r in bm25_results), default=1.0)
        
        # Process vector results
        for result in vector_results:
            doc_id = result["id"]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "score": 0,
                    "count": 0,
                    "metadata": result["metadata"],
                    "namespace": result.get("namespace", "unknown"),
                    "sources": [],
                    "search_methods": [],
                    "original_scores": {}
                }
            
            # Normalize score and add weighted contribution
            normalized_score = result["score"] / vector_max
            doc_scores[doc_id]["score"] += normalized_score * vector_weight
            doc_scores[doc_id]["count"] += 1
            doc_scores[doc_id]["sources"].append("vector")
            doc_scores[doc_id]["search_methods"].append("semantic")
            doc_scores[doc_id]["original_scores"]["vector"] = result.get("score", 0)
        
        # Process BM25 results
        for result in bm25_results:
            doc_id = result["id"]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "score": 0,
                    "count": 0,
                    "metadata": result["metadata"],
                    "namespace": result.get("namespace", "unknown"),
                    "sources": [],
                    "search_methods": [],
                    "original_scores": {}
                }
            
            # Normalize score and add weighted contribution
            normalized_score = result["score"] / bm25_max
            doc_scores[doc_id]["score"] += normalized_score * bm25_weight
            doc_scores[doc_id]["count"] += 1
            doc_scores[doc_id]["sources"].append("bm25")
            doc_scores[doc_id]["search_methods"].append("lexical")
            doc_scores[doc_id]["original_scores"]["bm25"] = result.get("score", 0)
            doc_scores[doc_id]["original_scores"]["raw_bm25"] = result.get("raw_bm25_score", 0)
        
        # Apply CombMNZ (multiply by number of lists where document appears)
        for doc_id in doc_scores:
            doc_scores[doc_id]["score"] *= doc_scores[doc_id]["count"]
        
        # Create final results list
        results = []
        for doc_id, data in doc_scores.items():
            # Determine the source based on which methods contributed
            if "vector" in data["sources"] and "bm25" in data["sources"]:
                source = "hybrid_combmnz"
                search_method = "hybrid"
            elif "vector" in data["sources"]:
                source = "vector"
                search_method = "semantic"
            else:
                source = "bm25"
                search_method = "lexical"
                
            result = {
                "id": doc_id,
                "score": data["score"],
                "metadata": data["metadata"],
                "namespace": data["namespace"],
                "source": source,
                "search_method": search_method,
                "contributing_sources": data["sources"],
                "original_scores": data["original_scores"]
            }
            results.append(result)
            print(f"CombMNZ fusion result: {doc_id}, score: {data['score']:.4f}, source: {source}")
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = None, fusion_method: str = None):
        """
        Perform hybrid search combining semantic and BM25 search.
        """
        # Start overall timing
        start_time = time.time()
        
        if alpha is None:
            alpha = self.alpha
            
        if fusion_method is None:
            fusion_method = self.fusion_method
        
        # Get enough results for effective fusion (3 times more than needed)
        search_k = top_k * 3
            
        # TIMING: Vector search    
        vector_start = time.time()
        vector_results = self._semantic_search(query, search_k)
        vector_time = time.time() - vector_start
        print(f"Vector search completed in {vector_time:.4f}s")
            
        # TIMING: BM25 search
        bm25_start = time.time()
        bm25_results = self._bm25_search(query, search_k)
        bm25_time = time.time() - bm25_start
        print(f"BM25 search completed in {bm25_time:.4f}s")
        
        print(f"\nVector search returned {len(vector_results)} results")
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
        
        # Remove duplicates before fusion
        # Create a set of document IDs from vector results
        vector_ids = {result["id"] for result in vector_results}
        
        # Filter BM25 results to remove any documents already in vector results
        unique_bm25_results = []
        duplicate_count = 0
        
        for result in bm25_results:
            if result["id"] in vector_ids:
                duplicate_count += 1
                # For duplicates, we'll keep the vector result but enhance it with BM25 info
                for v_result in vector_results:
                    if v_result["id"] == result["id"]:
                        v_result["bm25_score"] = result["score"]
                        v_result["raw_bm25_score"] = result.get("raw_bm25_score", 0)
                        v_result["source"] = "hybrid_duplicate"
                        break
            else:
                unique_bm25_results.append(result)
        
        print(f"Found {duplicate_count} duplicate documents between vector and BM25 results")
        
        # TIMING: Result fusion
        fusion_start = time.time()
        
        # Apply the selected fusion method
        if fusion_method == "rrf":
            results = self._reciprocal_rank_fusion(vector_results, unique_bm25_results)
            fusion_method_name = "RRF"
        elif fusion_method == "combmnz":
            results = self._combmnz_fusion(vector_results, unique_bm25_results, alpha, 1-alpha)
            fusion_method_name = "CombMNZ"
        else:
            # Default to weighted fusion
            fusion_method_name = "Weighted"
            print("Using weighted fusion method")
            result_scores = {}
            result_data = {}
            
            # Process vector results
            for result in vector_results:
                doc_id = result["id"]
                result_scores[doc_id] = alpha * result["score"]
                result_data[doc_id] = {
                    "id": result["id"],
                    "score": result["score"],
                    "metadata": result["metadata"],
                    "namespace": result.get("namespace", "unknown"),
                    "source": "hybrid_duplicate" if result.get("source") == "hybrid_duplicate" else result.get("source", "vector"),
                    "search_method": result.get("search_method", "semantic"),
                    "contributing_sources": ["vector"],
                    "original_scores": {"vector": result["score"]}
                }
                
                # If this was a duplicate that has BM25 info added
                if "bm25_score" in result:
                    result_scores[doc_id] += (1 - alpha) * result["bm25_score"]
                    result_data[doc_id]["contributing_sources"].append("bm25")
                    result_data[doc_id]["original_scores"]["bm25"] = result["bm25_score"]
                    result_data[doc_id]["original_scores"]["raw_bm25"] = result.get("raw_bm25_score", 0)
            
            # Process BM25 results and merge scores
            for result in unique_bm25_results:
                doc_id = result["id"]
                if doc_id in result_scores:
                    result_scores[doc_id] += (1 - alpha) * result["score"]
                    result_data[doc_id]["source"] = "hybrid_weighted"
                    result_data[doc_id]["search_method"] = "hybrid"
                    result_data[doc_id]["contributing_sources"].append("bm25")
                    result_data[doc_id]["original_scores"]["bm25"] = result["score"]
                    result_data[doc_id]["original_scores"]["raw_bm25"] = result.get("raw_bm25_score", 0)
                else:
                    result_scores[doc_id] = (1 - alpha) * result["score"]
                    result_data[doc_id] = {
                        "id": result["id"],
                        "score": result["score"],
                        "metadata": result["metadata"],
                        "namespace": result.get("namespace", "unknown"),
                        "source": result.get("source", "bm25"),
                        "search_method": result.get("search_method", "lexical"),
                        "contributing_sources": ["bm25"],
                        "original_scores": {
                            "bm25": result["score"],
                            "raw_bm25": result.get("raw_bm25_score", 0)
                        }
                    }
            
            # Sort by combined score
            sorted_ids = sorted(result_scores.keys(), key=lambda x: result_scores[x], reverse=True)
            
            # Create final results list
            results = []
            for doc_id in sorted_ids:
                if doc_id in result_data:
                    result = result_data[doc_id]
                    result["score"] = result_scores[doc_id]  # Update with combined score
                    results.append(result)
                    print(f"Weighted fusion result: {doc_id}, score: {result_scores[doc_id]:.4f}, source: {result['source']}")
            
        fusion_time = time.time() - fusion_start
        print(f"{fusion_method_name} fusion completed in {fusion_time:.4f}s")
        print(f"{fusion_method_name} fusion returned {len(results)} results")
        
        # Return empty list if no results after fusion
        if not results:
            print("No results found after fusion")
            return []
        
        # Calculate total search time
        total_time = time.time() - start_time
        print(f"\nTOTAL SEARCH TIME: {total_time:.4f}s")
        print(f"  - Vector: {vector_time:.4f}s ({vector_time/total_time*100:.1f}%)")
        print(f"  - BM25: {bm25_time:.4f}s ({bm25_time/total_time*100:.1f}%)")  
        print(f"  - Fusion: {fusion_time:.4f}s ({fusion_time/total_time*100:.1f}%)")
        print(f"  - Other: {total_time-vector_time-bm25_time-fusion_time:.4f}s")
        
        # Ensure diversity in the results - avoid repetitive content
        final_results = self._ensure_diverse_results(results[:top_k*2], top_k)
            
        return final_results
        
    def _ensure_diverse_results(self, results, top_k):
        """
        Ensure diversity in the final results by avoiding repetitive content.
        """
        if not results or len(results) <= 1:
            return results
            
        # Keep track of selected results and their text content
        selected_results = []
        seen_content = set()
        
        for result in results:
            # Extract the text content
            text = result.get("metadata", {}).get("text", "").strip()
            
            # Skip if no text content
            if not text:
                continue
                
            # Create a simplified representation of the text to detect near-duplicates
            # Use the first 100 characters as a fingerprint
            text_fingerprint = text[:100].lower()
            
            # Check if we've seen very similar content before
            is_duplicate = False
            for seen in seen_content:
                # Simple similarity check - if fingerprints share 80% of characters
                if len(set(text_fingerprint) & set(seen)) / len(set(text_fingerprint) | set(seen)) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                selected_results.append(result)
                seen_content.add(text_fingerprint)
                
                # Stop once we have enough results
                if len(selected_results) >= top_k:
                    break
        
        # If we couldn't find enough diverse results, fall back to the original list
        if len(selected_results) < min(top_k, len(results)):
            print(f"Warning: Could only find {len(selected_results)} diverse results, falling back to original ranking")
            return results[:top_k]
            
        return selected_results

    def _semantic_search(self, query: str, top_k: int = 5):
        """Perform vector search in Pinecone."""
        try:
            # Check embeddings cache first
            query_normalized = query.strip().lower()

            # OPTIMIZATION: Use cached embedding if available
            if query_normalized in self.query_embedding_cache:
                print("Using cached query embedding")
                query_embedding = self.query_embedding_cache[query_normalized]
            else:
                # Create embedding
                start_time = time.time()
                query_embedding = self.embedding_model.encode(query)
                encoding_time = time.time() - start_time
                print(f"Query encoding time: {encoding_time:.4f}s")

                # Normalize the embedding if needed
                if self.normalize_embeds:
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)

                # Cache the embedding
                if len(self.query_embedding_cache) >= self.max_cache_size:
                    # Remove a random item if cache is full
                    self.query_embedding_cache.pop(next(iter(self.query_embedding_cache)))
                self.query_embedding_cache[query_normalized] = query_embedding

            # Adapt dimension if needed
            query_embedding = self._adapt_dimension(query_embedding)

            all_results = []

            # Search in each namespace
            for namespace in self.namespaces:
                try:
                    results = self.index.query(
                        vector=query_embedding.tolist(),
                        top_k=top_k,
                        namespace=namespace,
                        include_metadata=True
                    )

                    # Extract matches
                    for match in results.matches:
                        result = {
                            "id": match.id,
                            "score": match.score,
                            "metadata": match.metadata,
                            "namespace": namespace,
                            "source": "vector",
                            "search_method": "semantic"
                        }
                        all_results.append(result)
                        print(f"Vector result: {match.id}, score: {match.score:.4f}")
                except Exception as e:
                    print(f"Error searching namespace {namespace}: {e}")

            # Sort by score
            all_results.sort(key=lambda x: x["score"], reverse=True)

            # Return top results
            print(f"Vector search returned {len(all_results)} results")
            return all_results[:top_k]
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

# For direct testing
if __name__ == "__main__":
    test_data = {
        "query": "What are the eligibility criteria for electric vehicle subsidies in Chandigarh?",
        "pineconeApiKey": "your_pinecone_api_key",
        "pineconeIndex": "your_pinecone_index",
        "top_k": 5,
        "alpha": 0.6,
        "fusion_method": "weighted"
    }
    
    print(execute(test_data)) 