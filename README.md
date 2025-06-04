# Hybrid Search RAG System

A clean, deployment-ready version of the hybrid search system that combines vector-based semantic search with BM25 lexical search.

## Files

- `hybrid_search_server_fixed.py` - Main server file
- `hybrid_search_for_workflow11.py` - Core hybrid search implementation
- `optimized_hybrid_search.py` - Optimized version of the hybrid search
- `embedding_cache_manager.py` - Manages embedding caching
- `embedding_cache.py` - Embedding cache implementation
- `namespace_manager.py` - Handles namespace management
- `parallel_search.py` - Implements parallel search capabilities
- `llm_cache_wrapper.py` - LLM caching wrapper
- `simple_optimizations.py` - Simple optimizations for search
- `config.py` - Configuration file (requires API keys)
- `hybrid_search_frontend.html` - Frontend UI

## Setup

1. Update the `config.py` file with your API keys
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `python hybrid_search_server_fixed.py`

## Deployment on Railway

This repository is configured for deployment on Railway. The `railway.json` and `Procfile` files are included for this purpose.

## Features

- **Hybrid Search**: Combines dense vector embeddings with BM25 lexical search
- **Multiple Fusion Methods**: Weighted fusion, Reciprocal Rank Fusion (RRF), and CombMNZ
- **Streaming Responses**: Real-time streaming of search results and LLM responses
- **Conversation History**: Maintains context across multiple queries
- **Optimized Performance**: Includes caching and parallel search capabilities

## API Endpoints

- `GET /`: Serves the frontend HTML
- `POST /api/search`: Performs a search and returns results
- `POST /api/search/stream`: Streams search results and LLM responses
- `POST /api/greeting`: Returns a simple greeting message

## Architecture

The system follows a hybrid search approach that combines vector-based semantic search with BM25 lexical search:

1. **Query Processing**: Analyzes and classifies the query
2. **Parallel Search**: Performs vector search and BM25 search concurrently
3. **Result Fusion**: Combines results using the specified fusion method
4. **Response Generation**: Generates a response using an LLM (Groq)

## Project Structure

- `Dockerfile`: Container configuration
- `hybrid-search-docker-compose.yml`: Docker Compose configuration
- `run_server.py`: Minimal server implementation
- `hybrid_search_frontend.html`: Frontend UI
- `requirements.txt`: Python dependencies

## API Keys

This application requires the following API keys:

- Pinecone API Key
- Groq API Key

## Customization

To customize the application, you can modify the following files:

- `hybrid_search_frontend.html`: Frontend UI
- `run_server.py`: Server implementation

## Troubleshooting

If you encounter any issues:

1. Check the logs: `docker-compose logs`
2. Verify your API keys are correctly set
3. Ensure all dependencies are installed 