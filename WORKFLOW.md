# Government Policy RAG System - Workflow

This document outlines the workflow and architecture of the Government Policy RAG system with optimized hybrid search capabilities.

## System Architecture

The system is built around a hybrid search approach that combines vector-based semantic search with BM25 lexical search. The architecture consists of several key components:

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│  Client Request   │────▶│  Hybrid Search    │────▶│  Response         │
│                   │     │  Server           │     │  Generation       │
└───────────────────┘     └─────────┬─────────┘     └───────────────────┘
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                          │  Optimized        │
                          │  Hybrid Search    │
                          │                   │
                          └─────────┬─────────┘
                                    │
                 ┌─────────────────┬┴────────────────┐
                 │                 │                 │
    ┌────────────▼─────┐ ┌─────────▼────────┐ ┌─────▼────────────┐
    │                  │ │                  │ │                  │
    │ Embedding Cache  │ │ Namespace        │ │ Parallel         │
    │ Manager          │ │ Manager          │ │ Search           │
    │                  │ │                  │ │                  │
    └────────────┬─────┘ └─────────┬────────┘ └─────┬────────────┘
                 │                 │                │
                 │                 │                │
    ┌────────────▼─────────────────▼────────────────▼────────────┐
    │                                                            │
    │                     Vector Database                        │
    │                     (Pinecone)                             │
    │                                                            │
    └────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Query Processing

When a query is received:

1. The query is analyzed by the `NamespaceManager` to determine relevant namespaces
2. The query is classified into a query type (e.g., "process", "eligibility", "fee")
3. The `EmbeddingCacheManager` checks if the query embedding already exists in cache
4. If not cached, a new embedding is generated and stored in cache

### 2. Parallel Search

The system performs two types of search in parallel:

1. **Vector Search**:
   - The query embedding is used to search the vector database
   - Only relevant namespaces are searched (filtered by the `NamespaceManager`)
   - Searches across namespaces are performed concurrently using the `ParallelSearcher`

2. **BM25 Search**:
   - The query text is used for lexical search using the BM25 algorithm
   - This finds documents with exact keyword matches

### 3. Result Fusion

The results from both search methods are combined using one of three fusion methods:

1. **Weighted Fusion**: Directly combines scores using a configurable alpha parameter
2. **Reciprocal Rank Fusion (RRF)**: Combines ranked lists with diminishing weights for lower ranks
3. **CombMNZ**: Weights normalized scores and rewards documents appearing in multiple result sets

### 4. Response Generation

The fused results are used to generate a response using an LLM (either OpenAI or Groq).

## Optimization Workflow

The system includes several optimizations to improve performance:

### Embedding Caching

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│               │     │               │     │               │
│  Query        │────▶│  Cache        │────▶│  Return       │
│               │     │  Lookup       │     │  Cached       │
└───────────────┘     └───────┬───────┘     │  Embedding    │
                              │             │               │
                              │ Not Found   └───────────────┘
                              ▼
                      ┌───────────────┐     ┌───────────────┐
                      │               │     │               │
                      │  Generate     │────▶│  Store in     │
                      │  Embedding    │     │  Cache        │
                      │               │     │               │
                      └───────────────┘     └───────────────┘
```

### Namespace Filtering

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│               │     │               │     │               │
│  Query        │────▶│  Extract      │────▶│  Match with   │
│               │     │  Keywords     │     │  Namespaces   │
└───────────────┘     └───────────────┘     └───────┬───────┘
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │               │
                                            │  Return       │
                                            │  Relevant     │
                                            │  Namespaces   │
                                            │               │
                                            └───────────────┘
```

### Parallel Search

```
                      ┌───────────────┐
                      │               │
                      │  Query        │
                      │               │
                      └───────┬───────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
  ┌───────────▼───────────┐     ┌─────────────▼─────────────┐
  │                       │     │                           │
  │  Thread Pool          │     │  Thread Pool              │
  │  Vector Search        │     │  BM25 Search              │
  │                       │     │                           │
  └───────────┬───────────┘     └───────────────┬───────────┘
              │                                 │
              └───────────────┬─────────────────┘
                              │
                      ┌───────▼───────┐
                      │               │
                      │  Combine      │
                      │  Results      │
                      │               │
                      └───────────────┘
```

## Performance Optimization Cycle

The system follows this cycle for continuous performance improvement:

1. **Measure**: Run `test_optimized_search.py` to establish baseline performance
2. **Optimize**: Implement optimizations in the core components
3. **Test**: Re-run tests to measure improvement
4. **Deploy**: Update the server with `update_server.py`
5. **Monitor**: Track performance in production
6. **Iterate**: Identify bottlenecks and return to step 2

## Deployment Workflow

To deploy the optimized system:

1. Run `update_server.py` to modify the server implementation
2. Start the server with `hybrid_search_server.py`
3. The server will use the optimized implementation with all performance enhancements

## Maintenance Workflow

For ongoing maintenance:

1. **Cache Management**: Periodically clean up the embedding cache if it grows too large
2. **Performance Monitoring**: Run performance tests regularly to ensure optimizations are effective
3. **Model Updates**: Update embedding models as needed for improved accuracy
4. **Namespace Updates**: Update namespace keywords as document collections evolve
