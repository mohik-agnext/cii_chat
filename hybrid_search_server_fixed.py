#!/usr/bin/env python3
from flask import Flask, request, jsonify, send_file, session, send_from_directory, Response
from simple_optimizations import SimpleOptimizer
from flask_cors import CORS
import asyncio
from hybrid_search_for_workflow11 import HybridSearch
import os
import json
import requests
import uuid
import sys
from datetime import datetime
import config
import groq
import re
import time
import threading
from optimized_hybrid_search import OptimizedHybridSearch
from flask_cors import CORS
import hashlib
from functools import lru_cache
import concurrent.futures

# Set environment variable to disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add these new utility functions for improved retrieval


def classify_question_type(query):
    """
    Classify the question type to optimize retrieval.
    """
    query_lower = query.lower()

    # Define patterns for different question types
    eligibility_patterns = [
        'eligibility',
        'qualify',
        'eligible',
        'criteria',
        'who can',
        'requirements']
    process_patterns = [
        'process',
        'procedure',
        'how to',
        'steps',
        'apply',
        'application',
        'submit']
    fee_patterns = [
        'fee',
        'cost',
        'price',
        'charge',
        'payment',
        'expense',
        'how much']
    penalty_patterns = [
        'penalty',
        'fine',
        'violation',
        'punish',
        'consequence']

    if any(pattern in query_lower for pattern in eligibility_patterns):
        return "eligibility"
    elif any(pattern in query_lower for pattern in process_patterns):
        return "process"
    elif any(pattern in query_lower for pattern in fee_patterns):
        return "fee"
    elif any(pattern in query_lower for pattern in penalty_patterns):
        return "penalty"
    else:
        return "general"


def compress_context(query, results, max_length=6000):
    """
    Extract most relevant parts of results based on query, preserving document structure
    and ensuring important content isn't lost.
    """
    # Short-circuit if already under max length
    total_length = sum(len(r.get('metadata', {}).get('text', ''))
                       for r in results)
    if total_length <= max_length:
        return results

    # Process query for better matching
    query_terms = set(query.lower().split())

    # Calculate available length per result (proportional allocation)
    result_count = len(results)
    if result_count == 0:
        return results

    # Identify important section headers and patterns to preserve
    important_patterns = [
        "eligibility criteria",
        "requirements",
        "incentives",
        "benefits",
        "application process",
        "fee structure",
        "penalties",
        "section",
        "chapter"]

    # Process each result
    compressed_results = []
    for result in results:
        metadata = result.get('metadata', {})
        text = metadata.get('text', '')

        # Skip if text is already short
        if len(text) < 500:
            compressed_results.append(result)
            continue

        # Split into sentences
        sentences = [
            s.strip() for s in text.replace(
                '\n', ' ').split('. ') if s.strip()]

        # Score sentences by multiple factors:
        # 1. Query term matches
        # 2. Position in document (earlier = more important for headers/context)
        # 3. Contains important patterns
        # 4. Sentence length (prefer neither too short nor too long)
        sentence_scores = []

        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()

            # 1. Base score from term matching
            term_match_score = sum(
                1 for term in query_terms if term in sentence_lower)

            # 2. Position score (first 3 sentences get bonus)
            position_score = 3.0 if i < 3 else (1.0 if i < 10 else 0)

            # 3. Important pattern matching
            pattern_score = 0
            for pattern in important_patterns:
                if pattern in sentence_lower:
                    pattern_score += 2

            # 4. Length factor (prefer medium-length sentences between 10-30
            # words)
            words = len(sentence.split())
            length_score = 1.0 if 10 <= words <= 30 else (
                0.5 if words < 10 else 0.7)

            # Calculate final score with weights
            final_score = (
                (term_match_score * 3.0) +
                position_score +
                pattern_score +
                length_score
            )

            sentence_scores.append((i, final_score, len(sentence)))

        # Sort by score descending
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Calculate target length for this document (proportional allocation)
        doc_allocation = max(
            300,
            max_length //
            result_count)  # At least 300 chars

        # Select sentences until we reach allocation
        selected_indices = []
        selected_length = 0

        # First, include highest-scoring sentences regardless of position
        top_indices = []
        # Always include top 5 highest scoring
        for i, score, length in sentence_scores[:5]:
            if score > 1.0:  # Only if reasonably relevant
                top_indices.append(i)
                selected_length += length

        # Then add more sentences based on original order if we have space
        remaining_budget = doc_allocation - selected_length
        if remaining_budget > 0:
            # Consider remaining sentences in document order
            remaining_scores = [(i, s, l) for i, s, l in sentence_scores
                                if i not in top_indices and s > 0]
            remaining_scores.sort(key=lambda x: x[0])  # Sort by position

            for i, score, length in remaining_scores:
                if selected_length + length <= doc_allocation:
                    selected_indices.append(i)
                    selected_length += length

        # Combine all selected indices and sort by original position
        all_indices = sorted(top_indices + selected_indices)

        # Reconstruct text with selected sentences, preserving order
        compressed_text = '. '.join([sentences[i]
                                    for i in all_indices if i < len(sentences)])
        if compressed_text and not compressed_text.endswith('.'):
            compressed_text += '.'

        # Update result
        result_copy = result.copy()
        result_copy['metadata'] = metadata.copy()
        result_copy['metadata']['text'] = compressed_text
        result_copy['metadata']['compressed'] = True
        result_copy['metadata']['compression_ratio'] = f"{
            len(compressed_text)}/{len(text)}"
        compressed_results.append(result_copy)

    return compressed_results


def filter_by_metadata(results, question_type):
    """
    Filter or boost results based on question type and metadata.
    """
    if not results:
        return results

    # Define section patterns for different question types
    pattern_map = {
        "eligibility": ['eligibility', 'criteria', 'requirements', 'qualification'],
        "process": ['process', 'procedure', 'application', 'steps', 'how to'],
        "fee": ['fee', 'cost', 'price', 'charge', 'payment', 'expenses'],
        "penalty": ['penalty', 'fine', 'violation', 'consequences']
    }

    patterns = pattern_map.get(question_type, [])

    if not patterns:
        return results

    # Boost scores for chunks with relevant section titles
    for result in results:
        metadata = result.get('metadata', {})
        section = metadata.get('section', '').lower()

        # Check if any pattern matches in the section title
        if any(pattern in section for pattern in patterns):
            # Boost the score
            result['score'] *= 1.5
            result['boosted'] = True

    # Re-sort based on updated scores
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


def rerank_results(query, results, top_k=5):
    """
    Re-rank search results using multiple relevance signals to improve accuracy.

    Args:
        query: The user query
        results: The list of search results to re-rank
        top_k: Maximum number of results to return

    Returns:
        Re-ranked list of results
    """
    if not results:
        return results

    query_terms = set(query.lower().split())

    # Extract question features
    query_lower = query.lower()
    is_specific = any(
        x in query_lower for x in [
            "specific",
            "exact",
            "precise",
            "definition",
            "what is",
            "what are"])
    is_process = any(
        x in query_lower for x in [
            "how to",
            "process",
            "steps",
            "procedure",
            "submit",
            "apply"])
    is_eligibility = any(
        x in query_lower for x in [
            "eligible",
            "qualify",
            "criteria",
            "requirement",
            "who can"])
    is_comparison = any(
        x in query_lower for x in [
            "compare",
            "difference",
            "versus",
            "vs",
            "better"])

    # Identify entities in query
    entity_patterns = [
        "MSME",
        "EV",
        "electric vehicle",
        "startup",
        "business",
        "industrial",
        "incentive"]
    query_entities = [
        entity for entity in entity_patterns if entity.lower() in query_lower]

    # Create reranking scores
    reranked_results = []

    for result in results:
        metadata = result.get('metadata', {})
        text = metadata.get('text', '')
        document_title = metadata.get('document_title', '')
        section = metadata.get('section', '')

        # Start with original score
        base_score = result.get('score', 0)

        # Initialize reranking score
        rerank_score = base_score

        # Get document type/category
        document_type = "unknown"
        if "policy" in document_title.lower():
            document_type = "policy"
        elif "application" in document_title.lower():
            document_type = "application"
        elif "guideline" in document_title.lower():
            document_type = "guideline"

        # 1. Exact match signals
        exact_match_score = 0
        text_lower = text.lower()

        # Exact query match in text
        if query_lower in text_lower:
            exact_match_score += 1.5

        # Title relevance
        if any(term in document_title.lower() for term in query_terms):
            exact_match_score += 1.0

        # Section relevance
        if any(term in section.lower() for term in query_terms):
            exact_match_score += 0.8

        # 2. Content quality signals
        quality_score = 0

        # Prefer longer content (more comprehensive) up to a point
        text_length = len(text)
        if 200 <= text_length <= 1000:
            quality_score += 0.3
        elif text_length > 1000:
            quality_score += 0.2

        # Prefer well-structured content
        has_structure = False
        if ":" in text or ";" in text or "-" in text:
            quality_score += 0.2
            has_structure = True

        # 3. Document type match signals
        document_match_score = 0

        # Match document type to query intent
        if is_process and document_type in ["application", "guideline"]:
            document_match_score += 1.2
        elif is_eligibility and document_type == "policy":
            document_match_score += 1.0
        elif is_specific and has_structure:
            document_match_score += 0.8

        # 4. Entity match bonus
        entity_match_score = 0
        for entity in query_entities:
            if entity.lower() in text_lower or entity.lower() in document_title.lower():
                entity_match_score += 0.7

        # 5. Special case handling
        special_case_score = 0

        # Specific EV policy questions
        if "electric vehicle" in query_lower and "electric vehicle policy" in document_title.lower():
            special_case_score += 1.5

        # MSME questions
        if "msme" in query_lower and "msme" in document_title.lower():
            special_case_score += 1.5

        # Calculate final reranking score with appropriate weights
        rerank_score = (
            base_score * 1.0 +  # Original score (vector or hybrid)
            exact_match_score * 1.5 +  # Exact matching is very important
            quality_score * 0.8 +  # Content quality matters but less than relevance
            document_match_score * 1.2 +  # Document type matching is important
            entity_match_score * 1.0 +  # Entity matching is important
            special_case_score * 1.5  # Special cases get highest weight
        )

        # Add entry to reranked results
        rerank_data = result.copy()
        rerank_data['original_score'] = base_score
        rerank_data['rerank_score'] = rerank_score
        rerank_data['rerank_components'] = {
            'exact_match': exact_match_score,
            'quality': quality_score,
            'document_match': document_match_score,
            'entity_match': entity_match_score,
            'special_case': special_case_score
        }
        reranked_results.append(rerank_data)

    # Sort by rerank score
    reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)

    # Take top-k
    reranked_results = reranked_results[:top_k]

    # Update scores to rerank scores
    for result in reranked_results:
        result['score'] = result['rerank_score']

    return reranked_results


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Progressive search route will be registered after search system
# initialization

# Initialize streaming routes
streaming_routes = None  # Will be initialized after search system is ready

app.secret_key = os.urandom(24)  # Secret key for sessions

# Dictionary to store conversation history by session ID
session_history = {}

# Global searcher instance
searcher = None
initializing = False

# Global variables for configuration
groq_api_key = None
groq_model = None
default_alpha = 0.5
default_fusion_method = "rrf"

# Maximum number of previous messages to include in context (sliding window)
MAX_HISTORY_MESSAGES = 10

# Maximum tokens to allow in the prompt (approximate)
MAX_TOKENS = 4000  # Reduced from 8000 for faster responses

# Cache for search results and LLM responses
search_cache = {}
llm_cache = {}
MAX_CACHE_SIZE = 100


def get_cache_key(query, params=None):
    """Generate a cache key from query and optional parameters."""
    key = query.lower().strip()
    if params:
        key += str(params)
    return hashlib.md5(key.encode()).hexdigest()


def add_to_cache(cache_dict, key, value):
    """Add a value to the cache with management of cache size."""
    cache_dict[key] = value
    # Manage cache size
    if len(cache_dict) > MAX_CACHE_SIZE:
        # Remove oldest item (first key)
        cache_dict.pop(next(iter(cache_dict)))


def parallel_search(query, top_k=5, alpha=0.5, fusion_method="rrf"):
    """Perform vector and BM25 search in parallel for faster results."""
    searcher = get_searcher()

    # Check cache first
    cache_key = get_cache_key(
        query, {"top_k": top_k, "alpha": alpha, "fusion": fusion_method})
    if cache_key in search_cache:
        print("Using cached search results")
        return search_cache[cache_key]

    search_start = time.time()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start both searches in parallel
        vector_future = executor.submit(
            searcher._semantic_search, query, top_k)
        bm25_future = executor.submit(searcher._bm25_search, query, top_k)

        # Wait for both to complete
        vector_results = vector_future.result()
        bm25_results = bm25_future.result()

    # Perform fusion based on method
    if fusion_method == "rrf":
        results = searcher._reciprocal_rank_fusion(
            vector_results, bm25_results)
    elif fusion_method == "combmnz":
        results = searcher._combmnz_fusion(vector_results, bm25_results)
    else:  # weighted
        results = searcher._weighted_fusion(
            vector_results, bm25_results, alpha)

    search_time = time.time() - search_start
    print(f"Parallel search completed in {search_time:.2f}s")

    # Cache the results
    add_to_cache(search_cache, cache_key, results)

    return results

# Load configuration


def load_config():
    try:
        from config import (
            PINECONE_API_KEY,
            PINECONE_INDEX,
            GROQ_API_KEY,
            GROQ_MODEL,
            DEFAULT_ALPHA,
            DEFAULT_FUSION_METHOD,
            EMBEDDING_MODEL
        )

        # Return the loaded values
        return (
            PINECONE_API_KEY,
            PINECONE_INDEX,
            GROQ_API_KEY,
            GROQ_MODEL,
            DEFAULT_ALPHA,
            DEFAULT_FUSION_METHOD,
            EMBEDDING_MODEL
        )
    except ImportError:
        print("Error: Please create a config.py file with the required settings")
        return None, None, None, None, None, None, None

# Initialize the search system in a background thread


def initialize_search_system():
    global searcher, initializing, groq_api_key, groq_model, default_alpha, default_fusion_method

    if searcher is not None or initializing:
        return

    initializing = True

    try:
        # Load configuration values
        loaded_pinecone_api_key, loaded_pinecone_index, loaded_groq_api_key, loaded_groq_model, \
            loaded_default_alpha, loaded_default_fusion_method, loaded_embedding_model = load_config()

        # Assign to global variables
        groq_api_key = loaded_groq_api_key
        groq_model = loaded_groq_model
        default_alpha = loaded_default_alpha
        default_fusion_method = loaded_default_fusion_method

        if loaded_pinecone_api_key and loaded_pinecone_index:
            print("-" * 50)
            print(f"Starting Hybrid Search Server with configuration:")
            print(f"• Embedding Model: {loaded_embedding_model}")
            print(f"• Pinecone Index: {loaded_pinecone_index}")
            print(f"• Fusion Method: {loaded_default_fusion_method}")
            print(f"• Alpha (vector vs BM25): {loaded_default_alpha}")
            print(f"• LLM Provider: Groq with model {groq_model}")
            print("-" * 50)

            # Initialize with time tracking
            start_time = time.time()

            from hybrid_search_for_workflow11 import HybridSearch
            searcher = HybridSearch(
                pinecone_api_key=loaded_pinecone_api_key,
                pinecone_index=loaded_pinecone_index,
                embedding_model=loaded_embedding_model,
                alpha=loaded_default_alpha,
                fusion_method=loaded_default_fusion_method
            )

            init_time = time.time() - start_time
            print(f"Hybrid search initialized in {init_time:.2f} seconds")
        else:
            print("Error: Missing required configuration")
    except Exception as e:
        print(f"Error initializing search system: {str(e)}")
    finally:
        initializing = False

# Helper function to ensure searcher is initialized


def get_searcher():
    global searcher, initializing

    if searcher is not None:
        return searcher

    # If initialization hasn't started yet, start it
    if not initializing:
        threading.Thread(target=initialize_search_system).start()

    # Wait for initialization with timeout
    wait_start = time.time()
    while searcher is None:
        if time.time() - wait_start > 120:  # 120 second timeout (increased from 30)
            raise Exception(
                "Searcher initialization timeout. Please try again later.")
        time.sleep(0.5)
        # Print progress message every 10 seconds
        if searcher is None and (time.time() - wait_start) % 10 < 0.5:
            print(f"Still waiting for searcher initialization... ({
                  int(time.time() - wait_start)} seconds elapsed)")
            sys.stdout.flush()

    return searcher


def get_or_create_session(session_id=None):
    """Get existing session ID or create a new one."""
    if session_id and session_id in session_history:
        return session_id

    # Create new session ID if none provided or invalid
    new_session_id = session_id or str(uuid.uuid4())

    # Initialize session history if needed
    if new_session_id not in session_history:
        print(f"Creating new session history for session_id: {new_session_id}")
        session_history[new_session_id] = []

    return new_session_id


def save_conversation(session_id, query, answer=None, metadata=None):
    """Save a conversation exchange to session history."""
    print(f"\n===== SAVING CONVERSATION =====\nSession: {session_id}\nQuery: {query[:50]}..." if len(
        query) > 50 else f"\n===== SAVING CONVERSATION =====\nSession: {session_id}\nQuery: {query}")
    print(f"Answer: {answer[:50]}..." if answer and len(
        answer) > 50 else f"Answer: {answer}")

    # Skip if no session ID
    if not session_id:
        print("No session ID provided, skipping save")
        return

    # Initialize session history if needed
    if session_id not in session_history:
        print(f"Creating new session history for session_id: {session_id}")
        session_history[session_id] = []

    # Check if this is an update to an existing query
    existing_entry = None
    if session_history[session_id]:
        # First check the last entry
        last_entry = session_history[session_id][-1]
        if last_entry.get('user') == query:
            existing_entry = last_entry
            print(f"Found matching last entry to update for query: {
                  query[:30]}...")
        else:
            # If not the last entry, check all entries for this query
            for entry in reversed(session_history[session_id]):
                if entry.get('user') == query:
                    existing_entry = entry
                    print(f"Found matching earlier entry to update for query: {
                          query[:30]}...")
                    break

    if existing_entry and answer is not None:
        # Update the existing entry with the new answer
        print(f"Updating existing entry with answer: {answer[:50]}..." if answer and len(
            answer) > 50 else f"Updating existing entry with answer: {answer}")
        existing_entry['answer'] = answer
        existing_entry['timestamp'] = datetime.now().isoformat()
        if metadata:
            existing_entry['metadata'] = metadata
        print(f"Updated existing conversation entry")
    else:
        # Create a new exchange record
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user": query,
            "answer": answer,
            "metadata": metadata or {}
        }

        # Add conversation exchange
        session_history[session_id].append(exchange)
        print(f"Added new conversation exchange with ID: {
              len(session_history[session_id])}")
        print(f"Exchange details: User query: '{query[:30]}...' Answer: '{
              answer[:30] if answer else None}...'")

    # Print the current conversation history for debugging
    print(f"Current conversation history for session {session_id}:")
    for i, entry in enumerate(session_history[session_id]):
        print(f"  {i + 1}. User: {entry.get('user',
                                            '')[:30]}... Answer: {entry.get('answer',
                                                                            '')[:30] if entry.get('answer') else 'None'}...")

    # Debug what was saved
    print(f"Conversation in session {session_id}:")
    print(f"  User query: {query[:50]}..." if len(
        query) > 50 else f"  User query: {query}")
    print(f"  Answer: {answer[:50]}..." if answer and len(
        answer) > 50 else f"  Answer: {answer}")
    print(f"  Total exchanges in session: {len(session_history[session_id])}")

    # Verify session was saved properly
    print(f"Session exists after save: {session_id in session_history}")
    print(f"Exchange count after save: {len(session_history[session_id])}")
    print(f"=====================================\n")

    # Limit history size per session
    if len(session_history[session_id]) > 10:
        session_history[session_id] = session_history[session_id][-10:]
        print(f"Trimmed session history to last 10 exchanges")


def get_conversation_history(session_id):
    """Get conversation history for a session."""
    if not session_id or session_id not in session_history:
        return []

    return session_history[session_id]

# Function to generate LLM response using Groq


def generate_llm_response(
        query,
        search_results,
        session_id,
        model=None,
        api_key=None,
        stream=False,
        include_history=True):
    # Save the query to conversation history immediately
    # This ensures the query is saved even if LLM generation fails
    save_conversation(session_id, query)
    # Use provided values or fall back to globals
    if model is None:
        model = groq_model
    if api_key is None:
        api_key = groq_api_key

    # If still None, try to load directly from config
    if not api_key or not model:
        try:
            from config import GROQ_API_KEY, GROQ_MODEL
            api_key = GROQ_API_KEY
            model = GROQ_MODEL
            print(f"Loaded Groq configuration directly from config.py")
        except ImportError:
            pass
    if not api_key or not model:
        return {"error": "Groq API key or model not configured"}

    # Format search results as context
    context = ""
    for i, result in enumerate(search_results):
        metadata = result.get('metadata', {})
        text = metadata.get('text', '')
        doc_title = metadata.get('document_title', 'Unknown document')
        section = metadata.get('section', '')

        # Format source info
        source_info = f"{doc_title}"
        if section:
            source_info += f" - {section}"

        # Add to context
        context += f"\n[Document {i + 1}] {source_info}\n{text}\n"

    # Add conversation context to the prompt if there's history
    system_message = """You are an expert assistant specializing in Chandigarh government policies and regulations."""

    # Create prompt for either case
    if include_history:
        prompt = f"""
CURRENT QUESTION: {
            query}

Based on the following context from Chandigarh government policy documents, please provide a comprehensive and accurate answer. Remember to consider the conversation history when relevant:

{context} """
    else:
        prompt = f"""
CURRENT QUESTION: {
            query}

Based on the following context from Chandigarh government policy documents, please provide a comprehensive and accurate answer:

{context} """

    # Call LLM API (Groq)
    try:
        client = groq.Client(api_key=api_key)

        if stream:
            # For streaming response
            completion_stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1200,
                top_p=0.9,
                stream=True  # Enable streaming
            )

            # For streaming, we need to collect the full response to save to conversation history
            # but we also need to return the stream object for streaming
            # We'll create a wrapper that captures the full response while
            # streaming
            full_response = []

            def stream_wrapper():
                nonlocal full_response
                try:
                    for chunk in completion_stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response.append(content)
                        yield chunk

                    # After streaming is complete, save the full response to
                    # conversation history
                    complete_response = ''.join(full_response)
                    print(
                        f"Stream completed. Full response length: {
                            len(complete_response)} chars")
                    print(f"First 100 chars of response: {
                          complete_response[:100]}...")

                    # Always save the conversation with the complete response
                    print(
                        f"Saving complete response to conversation history for session {session_id}")
                    save_conversation(
                        session_id=session_id,
                        query=query,
                        answer=complete_response,
                        metadata={
                            "search_results": [
                                {
                                    "id": r.get('id', ''),
                                    "score": r.get('score', 0),
                                    "document_title": r.get('metadata', {}).get('document_title', ''),
                                    "section": r.get('metadata', {}).get('section', '')
                                    # Just save top 3 for storage efficiency
                                } for r in search_results[:3]
                            ]
                        }
                    )
                except Exception as e:
                    print(f"Error in stream_wrapper: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

            # Return the wrapper generator
            return stream_wrapper()
        else:
            # For non-streaming response (original behavior)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1200,
                top_p=0.9
            )

            answer = completion.choices[0].message.content

            # Update the existing conversation entry with the answer
            # Get the most recent conversation for this session
            history = get_conversation_history(session_id)
            if history and history[-1].get('user') == query:
                # Update the existing entry
                print(
                    f"Updating existing conversation entry for session {session_id}")
                history[-1]['answer'] = answer
            else:
                # Create a new entry if needed
                print(
                    f"Creating new conversation entry for session {session_id}")
            save_conversation(session_id, query, answer)

            # Return the answer as a plain string for the llm_response field
            return answer
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return {"error": f"Failed to generate answer: {str(e)}"}


def perform_recursive_retrieval(
        query,
        initial_results,
        searcher,
        max_depth=1,
        max_total_results=20):
    """
    Perform recursive retrieval to find related policy sections.
    This helps ensure that all relevant requirements and criteria are captured.

    Args:
        query: Original user query
        initial_results: Initial search results
        searcher: HybridSearch instance
        max_depth: Maximum recursion depth
        max_total_results: Maximum total results to return

    Returns:
        Combined and deduplicated results
    """
    if not initial_results or max_depth <= 0:
        return initial_results

    # OPTIMIZATION: Check if this query really needs recursive retrieval
    # Only use recursive retrieval for policy and requirement related queries
    query_lower = query.lower()
    recursive_keywords = [
        'eligibility',
        'requirements',
        'criteria',
        'license',
        'permit',
        'procedure',
        'regulation',
        'compliance',
        'fee',
        'penalty']

    needs_recursive = any(
        keyword in query_lower for keyword in recursive_keywords)

    if not needs_recursive:
        print("Skipping recursive retrieval - query doesn't appear to need it")
        return initial_results

    all_results = initial_results.copy()
    seen_ids = {r.get('id') for r in initial_results if 'id' in r}

    # Extract policy-related section titles and key entities
    policy_sections = set()
    key_entities = set()

    # OPTIMIZATION: More targeted approach for extracting relevant terms
    # Extract the most important keywords from the query
    query_keywords = [
        word for word in query_lower.split() if len(word) > 3 and word not in [
            'what',
            'when',
            'where',
            'which',
            'how',
            'does',
            'that',
            'this']]

    for result in initial_results:
        metadata = result.get('metadata', {})
        section = metadata.get('section', '')
        text = metadata.get('text', '')

        # Add section title if it appears policy-related
        if section and any(keyword in section.lower()
                           for keyword in recursive_keywords):
            policy_sections.add(section)

        # Extract potential key entities with more targeted approach
        license_matches = re.findall(
            r'([L]-\d+[A-Z]?|license type [A-Z0-9-]+)', text, re.IGNORECASE)
        policy_matches = re.findall(
            r'([A-Z][a-z]+ (Policy|Act|Guidelines|Rules|Fee|Requirements))', text)

        for match in license_matches + [m[0] for m in policy_matches]:
            key_entities.add(match)

    # Generate follow-up queries based on extracted information
    follow_up_queries = []

    # OPTIMIZATION: More targeted query generation that combines keywords from original query
    # with discovered entities and sections

    # Add most important section-based query
    for section in policy_sections:
        # Combine section with original query keywords for more relevance
        section_query = f"{section} {' '.join(query_keywords[:3])}"
        if section_query not in follow_up_queries:
            follow_up_queries.append(section_query)

    # Add most important entity-based query
    for entity in key_entities:
        # Combine entity with specific requirement keywords
        requirement_terms = [
            term for term in recursive_keywords if term in query_lower]
        if not requirement_terms:
            requirement_terms = ['requirements', 'eligibility']
        entity_query = f"{entity} {' '.join(requirement_terms[:2])}"
        if entity_query not in follow_up_queries:
            follow_up_queries.append(entity_query)

    # OPTIMIZATION: Limit to top 2 follow-up queries instead of 3
    follow_up_queries = follow_up_queries[:2]

    # Skip further processing if no follow-up queries are generated
    if not follow_up_queries:
        return initial_results

    print(f"Generated {len(follow_up_queries)
                       } follow-up queries for recursive retrieval")

    # Perform follow-up searches
    for sub_query in follow_up_queries:
        print(f"Recursive retrieval - follow-up query: {sub_query}")

        sub_results = searcher.hybrid_search(
            query=sub_query,
            top_k=5,  # Use smaller top_k for follow-up queries
            alpha=0.3,  # Favor BM25 even more for follow-up queries
            fusion_method="rrf"  # Use RRF for better fusion
        )

        # Add new results that we haven't seen before
        for result in sub_results:
            result_id = result.get('id')
            if result_id and result_id not in seen_ids:
                seen_ids.add(result_id)
                # Mark as recursively retrieved
                result['metadata'] = result.get('metadata', {})
                result['metadata']['recursive_retrieval'] = True
                all_results.append(result)

    # Deduplicate and limit results
    # Sort by score to keep highest scoring results
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Limit total results
    return all_results[:max_total_results]


@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'hybrid_search_frontend.html')


@app.route('/api/search', methods=['POST'])
def search():
    # Initialize timing metrics dictionary
    timing_metrics = {}
    start_time = time.time()

    # Get query parameters
    if request.method == 'POST':
        data = request.get_json()
        query = data.get('query', '')
        session_id = data.get('session_id')
        top_k = data.get('top_k', 5)
        alpha = data.get('alpha', 0.7)
        fusion_method = data.get('fusion_method', 'rrf')
    else:
        # This should never happen with methods=['POST']
        return jsonify({"error": "Method not allowed"}), 405

    # Ensure we have a query
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Get the searcher instance
        searcher = get_searcher()

        # Perform the search
        search_start = time.time()
        results = searcher.hybrid_search(
            query, top_k=top_k, alpha=alpha, fusion_method=fusion_method)
        search_time = time.time() - search_start
        timing_metrics['search_time'] = search_time

        # Process results for response
        processed_results = []
        for result in results:
            processed_result = {
                "id": result.get("id", ""),
                "score": result.get("score", 0),
                "text": result.get("metadata", {}).get("text", ""),
                "source": result.get("source", "unknown"),
                "namespace": result.get("namespace", "")
            }
            processed_results.append(processed_result)

        # Generate response with LLM if requested
        generate_response = data.get('generate_response', False)
        llm_response = None
        if generate_response:
            llm_start = time.time()
            llm_result = generate_llm_response(query, results, session_id)
            timing_metrics['llm_time'] = time.time() - llm_start

            # Handle different response formats from generate_llm_response
            if isinstance(llm_result, dict) and 'answer' in llm_result:
                llm_response = llm_result['answer']
            elif isinstance(llm_result, dict) and 'error' in llm_result:
                llm_response = f"Error generating response: {
                    llm_result['error']}"
            else:
                # If it's a string or any other format, use it directly
                llm_response = llm_result

        # Calculate total time
        total_time = time.time() - start_time
        timing_metrics['total_time'] = total_time

        # Create response
        response = {
            "query": query,
            "results": processed_results,
            "timing": timing_metrics
        }

        if llm_response:
            response["llm_response"] = llm_response

        return jsonify(response)

    except Exception as e:
        print(f"Error in search: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add streaming search route


@app.route('/api/search/stream', methods=['GET', 'POST'])
def search_stream():
    # Initialize timing metrics dictionary
    timing_metrics = {}
    start_time = time.time()

    # Track if we should measure detailed timing metrics
    measure_step_times = False

    # Parse request data
    parse_start = time.time()

    # Handle both GET and POST requests
    if request.method == 'POST':
        data = request.json
        if not data:
            return Response(
                generate_error_stream("No JSON data provided"),
                mimetype='text/event-stream')
        query = data.get('query', '')
        session_id = data.get('session_id', '')
        include_history_param = data.get('include_history')
        include_history = True if include_history_param is None else (
            include_history_param or str(include_history_param).lower() == 'true')
        measure_step_times = data.get('measure_step_times', False)
    else:  # GET request
        query = request.args.get('query', '')
        session_id = request.args.get('session_id', '')
        include_history_param = request.args.get('include_history')
        include_history = True if include_history_param is None else (
            include_history_param.lower() == 'true')
        measure_step_times = request.args.get(
            'measure_step_times', 'false').lower() == 'true'

    print(f"Streaming request with query: '{query}', session_id: '{
          session_id}', include_history: {include_history}")

    if measure_step_times:
        timing_metrics['parse_request_time'] = time.time() - parse_start

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Get or create session ID
    session_start = time.time()
    session_id = get_or_create_session(session_id)

    if measure_step_times:
        timing_metrics['session_setup_time'] = time.time() - session_start

    # Debug session information
    print(f"\n======= SESSION DEBUG =======")
    print(f"Session ID: {session_id}")
    print(f"Session exists: {session_id in session_history}")
    print(f"Session history count: {len(session_history.get(session_id, []))}")

    # Print abbreviated history
    history_start = time.time()
    history = get_conversation_history(session_id)
    if history:
        print(f"Session history:")
        for i, exchange in enumerate(history):
            user_msg = exchange.get('user', '')
            user_msg = user_msg[:50] + \
                ('...' if len(user_msg) > 50 else '') if user_msg else ''
            answer = exchange.get('answer', '')
            answer = answer[:50] + ('...' if len(answer)
                                    > 50 else '') if answer else ''
            print(f"  {i + 1}. User: {user_msg}")
            print(f"     Bot: {answer}")
    print(f"==============================\n")

    if measure_step_times:
        timing_metrics['history_retrieval_time'] = time.time() - history_start

    # Save the user query before processing (without answer yet)
    # This ensures the query is saved even if processing fails
    save_start = time.time()
    save_conversation(session_id, query)

    if measure_step_times:
        timing_metrics['save_query_time'] = time.time() - save_start

    # Clean query of special characters
    preprocess_start = time.time()
    query = re.sub(r'[^\w\s?]', '', query).strip()

    if measure_step_times:
        timing_metrics['query_preprocessing_time'] = time.time() - \
            preprocess_start

    # Set default search parameters
    params_start = time.time()

    # Handle parameters differently based on request method
    if request.method == 'POST':
        top_k = int(data.get('top_k', 5))
        alpha = float(data.get('alpha', default_alpha))
        fusion_method = data.get('fusion_method', default_fusion_method)
        include_history = data.get('include_history', True)
    else:  # GET request
        top_k = int(request.args.get('top_k', 5))
        alpha = float(request.args.get('alpha', default_alpha))
        fusion_method = request.args.get(
            'fusion_method', default_fusion_method)
        include_history_param = request.args.get('include_history')
        include_history = True if include_history_param is None else (
            include_history_param.lower() == 'true')

    if measure_step_times:
        timing_metrics['params_setup_time'] = time.time() - params_start

    # Optional: Get conversation history for contextual search
    history_retrieval_start = time.time()
    conversation_history = []
    if include_history:
        conversation_history = get_conversation_history(session_id)

    if measure_step_times:
        timing_metrics['conversation_history_time'] = time.time() - \
            history_retrieval_start

    try:
        print(f"Processing query: '{query}'")

        # Classify query type
        classify_start = time.time()
        query_type = classify_question_type(query)
        print(f"Classified as: {query_type}")

        if measure_step_times:
            timing_metrics['query_classification_time'] = time.time() - \
                classify_start

        # Search with timing
        search_start = time.time()

        # Perform primary search
        vector_search_start = time.time()
        search_results = get_searcher().hybrid_search(
            query=query,
            top_k=top_k,
            alpha=alpha,
            fusion_method=fusion_method
        )

        if measure_step_times:
            timing_metrics['hybrid_search_time'] = time.time() - \
                vector_search_start

        # Apply metadata filtering based on question type
        filter_start = time.time()
        filtered_results = filter_by_metadata(search_results, query_type)

        if measure_step_times:
            timing_metrics['metadata_filtering_time'] = time.time() - \
                filter_start

        # Apply re-ranking to improve result ordering
        rerank_start = time.time()
        reranked_results = rerank_results(query, filtered_results, top_k=top_k)

        if measure_step_times:
            timing_metrics['initial_reranking_time'] = time.time() - \
                rerank_start

        search_time = time.time() - search_start
        if measure_step_times:
            timing_metrics['total_search_time'] = search_time

        print(
            f"Search completed in {
                search_time:.2f}s, found {
                len(reranked_results)} results")

        # Use recursive retrieval for policy questions
        if "policy" in query.lower() or query_type in [
                "eligibility", "process"]:
            recursive_start = time.time()
            combined_results = perform_recursive_retrieval(
                query, reranked_results, get_searcher())
            recursive_time = time.time() - recursive_start
            print(
                f"Recursive retrieval completed in {
                    recursive_time:.2f}s, total results: {
                    len(combined_results)}")
            # Re-rank again after recursive retrieval
            final_results = rerank_results(
                query, combined_results, top_k=top_k)
        else:
            final_results = reranked_results

        # Create context for LLM
        context = ""
        for i, result in enumerate(final_results):
            metadata = result.get('metadata', {})
            text = metadata.get('text', '')
            source = result.get('source', 'hybrid')
            doc_title = metadata.get('document_title', 'Unknown document')
            section = metadata.get('section', '')
            score = result.get('score', 0)

            # Format source info
            source_info = f"{doc_title}"
            if section:
                source_info += f" - {section}"

            # Add to context
            context += f"\n[Document {i + 1}] {source_info}\n{text}\n"
            if 'rerank_components' in result:
                context_debug = f"(Score: {
                    score:.4f}, Source: {source}, Exact Match: {
                    result['rerank_components']['exact_match']:.2f})"
                print(f"Result {i + 1}: {doc_title[:40]}... {context_debug}")
            else:
                print(
                    f"Result {i + 1}: {doc_title[:40]}... (Score: {score:.4f}, Source: {source})")

        # Compress context if needed
        if len(context) > MAX_TOKENS:
            compressed_results = compress_context(query, final_results)

            # Rebuild context with compressed results
            context = ""
            for i, result in enumerate(compressed_results):
                metadata = result.get('metadata', {})
                text = metadata.get('text', '')
                doc_title = metadata.get('document_title', 'Unknown document')
                section = metadata.get('section', '')

                source_info = f"{doc_title}"
                if section:
                    source_info += f" - {section}"

                context += f"\n[Document {i + 1}] {source_info}\n{text}\n"

        # Return a streaming response
        response_start = time.time()

        # Set up streaming response
        def stream_wrapper():
            # Stream the results using the generate_stream function
            try:
                # First yield metadata with session ID
                yield f"data: {json.dumps({'type': 'metadata', 'session_id': session_id})}\n\n"

                # Stream search results
                top_results = [
                    {
                        'id': r.get('id', ''),
                        'score': r.get('score', 0),
                        'document_title': r.get('metadata', {}).get('document_title', 'Unknown'),
                        'section': r.get('metadata', {}).get('section', ''),
                        'snippet': r.get('metadata', {}).get('text', '')[:100] + '...'
                    }
                    # Only send top 3 for efficiency
                    for r in final_results[:3]
                ]
                yield f"data: {json.dumps({'type': 'search_results', 'results': top_results})}\n\n"

                # Generate streaming response from LLM
                full_response = ""
                completion_stream = generate_llm_response(
                    query=query,
                    search_results=final_results,
                    session_id=session_id,
                    stream=True,  # Enable streaming
                    include_history=include_history
                )

                # Stream response chunks
                for chunk in completion_stream:
                    if hasattr(
                            chunk.choices[0].delta,
                            'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"

                # Save the full conversation after streaming is complete
                save_conversation(
                    session_id=session_id,
                    query=query,
                    answer=full_response,
                    metadata={
                        "search_results": [
                            {
                                "id": r.get('id', ''),
                                "score": r.get('score', 0),
                                "document_title": r.get('metadata', {}).get('document_title', ''),
                                "section": r.get('metadata', {}).get('section', '')
                                # Just save top 3 for storage efficiency
                            } for r in final_results[:3]
                        ],
                        "search_time": search_time,
                        "response_time": time.time() - response_start
                    }
                )

                # Signal completion
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                print(f"Error in stream_wrapper: {str(e)}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        # Return the streaming response
        return Response(stream_wrapper(), mimetype='text/event-stream')

    except Exception as e:
        print(f"Error processing search: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": f"An error occurred: {str(e)}",
            "session_id": session_id
        }), 500


@app.route('/api/search/stream/v2', methods=['GET', 'POST'])
def search_stream_v2():
    start_time = time.time()

    # Handle both GET and POST requests
    if request.method == 'POST':
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        query = data.get('query', '')
        session_id = data.get('session_id', '')
        top_k = int(data.get('top_k', 5))
        alpha = float(data.get('alpha', default_alpha))
        fusion_method = data.get('fusion_method', default_fusion_method)
    else:  # GET request
        query = request.args.get('query', '')
        session_id = request.args.get('session_id', '')
        top_k = int(request.args.get('top_k', 5))
        alpha = float(request.args.get('alpha', default_alpha))
        fusion_method = request.args.get(
            'fusion_method', default_fusion_method)

    if not query:
        return Response(
            generate_error_stream("No query provided"),
            mimetype='text/event-stream')

    # Get or create session ID
    session_id = get_or_create_session(session_id)

    # Clean query of special characters
    query = re.sub(r'[^\w\s?]', '', query).strip()

    # Check if it's a greeting
    greeting_words = [
        'hello',
        'hi',
        'hey',
        'greetings',
        'good morning',
        'good afternoon',
        'good evening']
    is_greeting = any(word in query.lower() for word in greeting_words)

    if is_greeting:
        return Response(
            generate_greeting_stream(
                session_id,
                query),
            mimetype='text/event-stream')

    try:
        print(f"Processing streaming query: '{query}'")
        query_type = classify_question_type(query)
        print(f"Classified as: {query_type}")

        # Search with timing
        search_start = time.time()

        # Perform primary search
        search_results = get_searcher().hybrid_search(
            query=query,
            top_k=top_k,
            alpha=alpha,
            fusion_method=fusion_method
        )

        # Apply metadata filtering based on question type
        filtered_results = filter_by_metadata(search_results, query_type)

        # Apply re-ranking to improve result ordering
        reranked_results = rerank_results(query, filtered_results, top_k=top_k)

        search_time = time.time() - search_start
        print(
            f"Search completed in {
                search_time:.2f}s, found {
                len(reranked_results)} results")

        # Use recursive retrieval for policy questions
        if "policy" in query.lower() or query_type in [
                "eligibility", "process"]:
            recursive_start = time.time()
            combined_results = perform_recursive_retrieval(
                query, reranked_results, get_searcher())
            recursive_time = time.time() - recursive_start
            print(
                f"Recursive retrieval completed in {
                    recursive_time:.2f}s, total results: {
                    len(combined_results)}")
            # Re-rank again after recursive retrieval
            final_results = rerank_results(
                query, combined_results, top_k=top_k)
        else:
            final_results = reranked_results

        # Print top results
        for i, result in enumerate(final_results[:3]):
            metadata = result.get('metadata', {})
            doc_title = metadata.get('document_title', 'Unknown document')
            score = result.get('score', 0)
            source = result.get('source', 'hybrid')

            if 'rerank_components' in result:
                context_debug = f"(Score: {
                    score:.4f}, Source: {source}, Exact Match: {
                    result['rerank_components']['exact_match']:.2f})"
                print(f"Result {i + 1}: {doc_title[:40]}... {context_debug}")
            else:
                print(
                    f"Result {i + 1}: {doc_title[:40]}... (Score: {score:.4f}, Source: {source})")

        # Create streaming response
        return Response(
            generate_stream(
                query,
                session_id,
                final_results,
                include_history=include_history),
            mimetype='text/event-stream')

    except Exception as e:
        print(f"Error processing streaming search: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response(
            generate_error_stream(
                str(e)),
            mimetype='text/event-stream')


def generate_greeting_stream(session_id=None, query=None):
    """Generate a streaming response for greeting messages."""
    full_response = ""

    # Make sure we have session_id
    if session_id is None:
        session_id = get_or_create_session()

    # Default query if none provided
    if query is None:
        query = "hello"

    # Stream response header
    yield f"data: {json.dumps({'type': 'metadata', 'session_id': session_id, 'is_greeting': True})}\n\n"

    # Create a friendly greeting
    greeting = "Hello! I'm your AI assistant for Chandigarh policies and regulations. How can I help you today?"

    # Stream the greeting character by character to simulate typing
    for char in greeting:
        full_response += char
        yield f"data: {json.dumps({'type': 'chunk', 'content': char})}\n\n"
        time.sleep(0.01)  # Small delay for realistic typing effect

    # Save the full response to history
    save_conversation(session_id, query, full_response)

    # Stream response footer
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


def generate_stream(query, session_id, final_results, include_history=True):
    """Generate a streaming response for search results."""
    full_response = ""

    # Debug conversation history
    history = get_conversation_history(session_id)
    if history:
        print(
            f"Found {
                len(history)} previous exchanges in conversation history for session {session_id}")
        for i, exchange in enumerate(history[-3:]):  # Show last 3 exchanges
            print(
                f"  Exchange {
                    i + 1}: User: {
                    exchange.get(
                        'user', '')[
                        :30]}... Answer: {
                        exchange.get(
                            'answer', '')[
                                :30]}...")
    else:
        print(f"No conversation history found for session {session_id}")

    # Stream metadata header
    yield f"data: {json.dumps({'type': 'metadata', 'session_id': session_id})}\n\n"

    # Stream the initial search results
    top_results = [
        {
            'id': r.get('id', ''),
            'score': r.get('score', 0),
            'document_title': r.get('metadata', {}).get('document_title', 'Unknown'),
            'section': r.get('metadata', {}).get('section', ''),
            'snippet': r.get('metadata', {}).get('text', '')[:100] + '...'
        }
        for r in final_results[:3]  # Only send top 3 for efficiency
    ]
    yield f"data: {json.dumps({'type': 'search_results', 'results': top_results})}\n\n"

    # Get streaming response from LLM
    completion_stream = generate_llm_response(
        query=query,
        search_results=final_results,
        session_id=session_id,
        stream=True,  # Enable streaming
        include_history=include_history  # Pass the include_history parameter
    )

    # Stream response chunks from LLM
    try:
        for chunk in completion_stream:
            if hasattr(
                    chunk.choices[0].delta,
                    'content') and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    # Note: We don't save conversation here because it's already saved in the
    # stream_wrapper

    # Stream completion message
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


def generate_error_stream(error_msg):
    """Generate a streaming response for errors."""
    yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"


if __name__ == '__main__':
    # Make sure any existing Python processes are killed
    try:
        import psutil
        import os
        import signal

        current_pid = os.getpid()

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['pid'] == current_pid:
                continue

            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'python' in proc.info['name'].lower(
                ) and 'hybrid_search_server.py' in ' '.join(cmdline):
                    print(f"Killing existing process: {proc.info['pid']}")
                    os.kill(proc.info['pid'], signal.SIGTERM)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except ImportError:
        print("psutil not available, skipping process cleanup")

    # Check if frontend file exists, if not warn the user
    if not os.path.exists('hybrid_search_frontend.html'):
        print("Warning: Frontend file 'hybrid_search_frontend.html' not found!")

    # Add error handler for server errors
    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": str(e)}), 500

    # Run the Flask app
    # Use port 8080 to avoid conflicts with macOS AirPlay (port 5000)
    port = 8080
    print(f"Starting server at http://localhost:{port}")

    # Start initialization in a background thread
    init_thread = threading.Thread(target=initialize_search_system)
    init_thread.daemon = True
    init_thread.start()

    # Start the Flask app with debug mode off to avoid reloading issues
    app.run(host='0.0.0.0', port=port, debug=False)
