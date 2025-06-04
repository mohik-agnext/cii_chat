# Hybrid Search (BM25 + Dense Vectors)

This document explains the hybrid search implementation that combines BM25 lexical search with dense vector search for improved retrieval accuracy in RAG systems.

## Overview

The hybrid search implementation combines the strengths of two different retrieval approaches:

1. **Dense Vector Search**: Uses embeddings to capture semantic meaning, good for finding conceptually related content even when exact keywords aren't present.

2. **BM25 Lexical Search**: Uses term frequency and inverse document frequency statistics to find exact keyword matches, good for precise terminology retrieval.

By combining these approaches, we achieve better retrieval that works well for both semantic similarity and exact keyword matching.

## Key Features

- **Multiple Fusion Methods**:
  - **Weighted Fusion**: Directly combines BM25 and vector scores using a configurable alpha parameter
  - **Reciprocal Rank Fusion (RRF)**: Combines ranked lists from both methods, with diminishing weights for lower ranks
  - **CombMNZ**: Weights normalized scores and rewards documents that appear in multiple result sets

- **Adaptive Dimension Handling**: Automatically adapts to any mismatch between embedding model dimension and vector database dimension

- **Cross-Encoder Reranking**: Optional second-stage reranking using more powerful bi-encoder models for improved accuracy

- **Interactive Parameter Tuning**: Find the optimal balance between lexical and semantic search for each query

## Implementation Details

### Fusion Methods

#### 1. Weighted Fusion

The simplest approach that combines scores from both methods using a weighted average:

```python
final_score = alpha * vector_score + (1 - alpha) * bm25_score
```

Where:
- `alpha` is the weight parameter (0-1) that determines the balance between methods
- 0 = BM25 only, 1 = vector only

#### 2. Reciprocal Rank Fusion (RRF)

A more sophisticated approach that combines ranked lists rather than raw scores:

```python
rrf_score = sum(1 / (rank + k))
```

Where:
- `k` is a constant (typically 60) that prevents items with very high ranks from dominating
- `rank` is the position in each ranked list

This method has been shown to be effective for web search and information retrieval tasks.

#### 3. CombMNZ

A method that normalizes scores and rewards documents that appear in multiple result sets:

```python
combmnz_score = sum(normalized_scores) * appearance_count
```

Where:
- `normalized_scores` are the scores from each method, scaled to 0-1
- `appearance_count` is the number of methods that return the document (1 or 2)

### When to Use Each Method

- **Weighted Fusion**: Best for simpler queries where you know the right balance of keyword vs semantic importance
- **RRF**: Best for complex queries with varied terminology or when relevance scores differ greatly between methods
- **CombMNZ**: Best for queries that benefit from evidence from multiple retrieval methods

## Code Structure

### Main Components

- **BM25 Initialization**: Samples documents from the vector database to build a BM25 index
- **Semantic Search**: Uses the embedding model to create query vectors and search the vector database
- **BM25 Search**: Uses the BM25 algorithm to find matching documents based on keywords
- **Fusion Methods**: Implementations of different fusion strategies
- **Cross-Encoder Reranking**: Optional reranking of fused results
- **Interactive Chat Loop**: Command-line interface with parameter tuning capabilities

### Key Functions

- `_initialize_bm25()`: Builds the BM25 index from document samples
- `_semantic_search()`: Performs vector search in Pinecone
- `_bm25_search()`: Performs lexical search using BM25
- `_reciprocal_rank_fusion()`: Implements RRF algorithm
- `_combmnz_fusion()`: Implements CombMNZ algorithm
- `hybrid_search()`: Main function that combines both search methods
- `tune_hybrid_weight()`: Automatically finds optimal parameters

## Usage

### Basic Usage

```python
python hybrid_bm25_rag.py
```

### With Specific Fusion Method

```python
python hybrid_bm25_rag.py --fusion rrf
```

### With Reranking

```python
python hybrid_bm25_rag.py --fusion combmnz --rerank
```

### With Custom Alpha Value

```python
python hybrid_bm25_rag.py --alpha 0.7
```

## Interactive Commands

During the chat session, you can use these commands:

- `/debug on|off` - Toggle debug information
- `/rewrite on|off` - Toggle query rewriting
- `/alpha [value]` - Set vector vs BM25 weight (0=BM25 only, 1=vector only)
- `/fusion [method]` - Set fusion method (weighted, rrf, combmnz)
- `/rerank on|off` - Toggle cross-encoder reranking
- `/tune [simple|full]` - Auto-tune parameters for current question

## Performance Considerations

- BM25 initialization adds a small startup cost but results in faster query time
- Fusion methods have minimal computational overhead
- Cross-encoder reranking is computationally expensive but significantly improves result quality
- For large document collections, sampling for BM25 index may need to be increased 