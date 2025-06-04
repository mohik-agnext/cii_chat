# Government Policy RAG System - Architecture

This document provides a detailed technical overview of the system architecture, components, and their interactions.

## System Components

### 1. Core Search Components

#### HybridSearch (Base Implementation)
The foundation of the search system, implemented in `hybrid_search_for_workflow11.py`. It combines:
- Vector-based semantic search using embeddings
- BM25 lexical search for keyword matching
- Multiple fusion methods to combine results

#### OptimizedHybridSearch
An enhanced version of the base implementation in `optimized_hybrid_search.py`, which adds:
- Embedding caching
- Namespace filtering
- Parallel search capabilities
- Performance monitoring

### 2. Optimization Components

#### EmbeddingCacheManager
Implemented in `embedding_cache_manager.py`, this component:
- Maintains both in-memory and disk-based caches for embeddings
- Uses LRU (Least Recently Used) eviction policy for memory cache
- Persists embeddings to disk for long-term storage
- Handles cache metadata management

#### NamespaceManager
Implemented in `namespace_manager.py`, this component:
- Maps keywords to relevant namespaces
- Classifies queries into different types
- Filters the search space to only relevant namespaces
- Improves search performance by reducing the number of namespaces to search

#### ParallelSearcher
Implemented in `parallel_search.py`, this component:
- Executes searches across multiple namespaces concurrently
- Runs vector and BM25 search in parallel
- Uses thread pools to manage concurrent execution
- Aggregates and ranks results from parallel searches

### 3. Server Components

#### HybridSearchServer
The main server implementation in `hybrid_search_server.py`, which:
- Provides REST API endpoints for search
- Handles request parsing and response formatting
- Manages authentication and rate limiting
- Integrates with LLM providers for response generation

#### UpdateServer
A utility in `update_server.py` that:
- Backs up the original server implementation
- Updates imports to use the optimized implementation
- Configures the optimized search with appropriate parameters
- Creates necessary cache directories

### 4. Testing and Evaluation

#### TestOptimizedSearch
Implemented in `test_optimized_search.py`, this component:
- Compares performance between original and optimized implementations
- Measures search times for vector, BM25, and hybrid search
- Calculates performance improvements
- Generates performance reports

## Technical Details

### Embedding Models

The system supports various embedding models through the Hugging Face Transformers library:
- Default: "BAAI/bge-large-en-v1.5"
- Dimension adaptation to match Pinecone index dimensions
- Normalization options for improved search quality

### Vector Database (Pinecone)

The system uses Pinecone as the vector database with these features:
- Multiple namespaces for organizing document collections
- Metadata filtering capabilities
- Hybrid search support
- Retry mechanism for handling connection issues

### BM25 Implementation

The BM25 implementation includes:
- Document sampling from the vector database
- Term frequency and inverse document frequency calculations
- Score normalization for fusion with vector search results
- Configurable parameters for tuning

### Fusion Methods

The system supports three fusion methods:

#### 1. Weighted Fusion
```python
final_score = alpha * vector_score + (1 - alpha) * bm25_score
```
Where `alpha` is a configurable weight parameter (0-1).

#### 2. Reciprocal Rank Fusion (RRF)
```python
rrf_score = sum(1 / (rank + k))
```
Where `k` is a constant (typically 60) and `rank` is the position in each ranked list.

#### 3. CombMNZ
```python
combmnz_score = sum(normalized_scores) * appearance_count
```
Where `normalized_scores` are the scores from each method, scaled to 0-1, and `appearance_count` is the number of methods that return the document.

### Caching Strategy

The embedding cache uses a two-tier approach:
1. **Memory Cache**: Fast access, limited size with LRU eviction
2. **Disk Cache**: Persistent storage using file system
3. **Metadata Tracking**: Records model, creation time, and dimensions

### Namespace Filtering Algorithm

The namespace filtering uses:
1. Keyword extraction from the query
2. Keyword-to-namespace mapping
3. Direct namespace matching
4. Score-based ranking of namespaces
5. Minimum namespace selection to ensure coverage

### Parallel Processing

The parallel processing implementation uses:
1. ThreadPoolExecutor for managing worker threads
2. Future-based asynchronous execution
3. Concurrent namespace searching
4. Parallel execution of vector and BM25 search
5. Result aggregation and error handling

## Data Flow Sequence

1. **Query Reception**:
   ```
   Client -> HybridSearchServer -> OptimizedHybridSearch
   ```

2. **Query Processing**:
   ```
   OptimizedHybridSearch -> NamespaceManager (classify query, filter namespaces)
   OptimizedHybridSearch -> EmbeddingCacheManager (get/create embedding)
   ```

3. **Search Execution**:
   ```
   OptimizedHybridSearch -> ParallelSearcher (execute searches)
   ParallelSearcher -> Pinecone (vector search)
   ParallelSearcher -> BM25Index (lexical search)
   ```

4. **Result Processing**:
   ```
   ParallelSearcher -> OptimizedHybridSearch (aggregate results)
   OptimizedHybridSearch -> FusionMethod (combine results)
   OptimizedHybridSearch -> HybridSearchServer (return results)
   ```

5. **Response Generation**:
   ```
   HybridSearchServer -> LLM Provider (generate response)
   HybridSearchServer -> Client (return response)
   ```

## Performance Considerations

### Time Complexity

- **Vector Search**: O(d * n) where d is embedding dimension and n is number of vectors
- **BM25 Search**: O(t * d) where t is number of terms and d is number of documents
- **Namespace Filtering**: O(k * n) where k is number of keywords and n is number of namespaces
- **Parallel Search**: Reduces time by factor of min(c, n) where c is number of cores and n is number of namespaces

### Space Complexity

- **Embedding Cache**: O(m * d) where m is number of cached queries and d is embedding dimension
- **BM25 Index**: O(t * d) where t is number of terms and d is number of documents
- **Namespace Mapping**: O(k * n) where k is number of keywords and n is number of namespaces

### Bottlenecks and Optimizations

1. **Embedding Generation**: Optimized with caching
2. **Vector Search**: Optimized with namespace filtering and parallel search
3. **BM25 Index**: Optimized with document sampling and term frequency caching
4. **Result Fusion**: Optimized with efficient fusion algorithms

## Extensibility Points

The architecture is designed to be extensible at these points:

1. **Embedding Models**: Can be replaced with any model compatible with the interface
2. **Vector Database**: Can be replaced with any vector database with similar capabilities
3. **Fusion Methods**: New fusion methods can be added by implementing the fusion interface
4. **Namespace Management**: Custom namespace filtering strategies can be implemented
5. **Caching Strategies**: Alternative caching implementations can be plugged in
