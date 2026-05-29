"""
Tool 2 – RAG Search (FAISS + Claude 3.5 Sonnet)
================================================
Flow:
  1. Embed the user query with Amazon Titan Embed v2.
  2. Search the local FAISS index for the top-k most similar chunks.
  3. Build a context string from retrieved chunks.
  4. Send context + query to Claude 3.5 Sonnet via Bedrock.
  5. Return the grounded answer + source references.

The FAISS index is built from existing S3 documents via Tool 1.
On first startup, if the index is empty, call build_index_from_s3()
to process all existing files in the S3 documents/ folder.
"""

from __future__ import annotations

import json
import os
from typing import Any

from utils import (
    AWS_REGION,
    LLM_MODEL_ID,
    S3_BUCKET,
    S3_DOCS_PREFIX,
    append_to_faiss,
    chunk_text,
    download_from_s3,
    embed_query,
    get_bedrock_client,
    list_s3_documents,
    make_doc_id,
    parse_with_markitdown,
    embed_texts,
    search_faiss,
)

RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))


# ── Build index from existing S3 data ────────────────────────────────────────

def build_index_from_s3() -> dict[str, Any]:
    """
    Process all existing documents in S3_DOCS_PREFIX and build/rebuild
    the FAISS index. Call this once to bootstrap from existing S3 data.

    Returns: {processed_files, total_chunks, total_vectors}
    """
    s3_objects = list_s3_documents()
    if not s3_objects:
        return {"processed_files": 0, "total_chunks": 0, "total_vectors": 0, "message": "No files found in S3."}

    processed = 0
    total_chunks = 0

    for obj in s3_objects:
        key = obj["key"]
        filename = key.split("/")[-1]
        if not filename or "." not in filename:
            continue  # skip folder markers

        try:
            file_bytes = download_from_s3(key)
            markdown_text = parse_with_markitdown(file_bytes, filename)
            chunks = chunk_text(markdown_text)
            if not chunks:
                continue

            # Use the S3 key path as a stable doc_id
            doc_id = make_doc_id(key)
            vectors = embed_texts(chunks)
            chunk_metas = [
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "s3_key": key,
                    "chunk_index": i,
                    "text": chunk,
                }
                for i, chunk in enumerate(chunks)
            ]
            total_vectors = append_to_faiss(vectors, chunk_metas)
            total_chunks += len(chunks)
            processed += 1
        except Exception as e:
            print(f"[build_index] Skipping {key}: {e}")
            continue

    return {
        "processed_files": processed,
        "total_chunks": total_chunks,
        "total_vectors": total_vectors if processed > 0 else 0,
    }


# ── RAG answer generation ─────────────────────────────────────────────────────

def _generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Call Claude 3.5 Sonnet with retrieved context to generate a grounded answer."""
    client = get_bedrock_client()

    context_text = "\n\n---\n\n".join(
        f"[Source: {c.get('filename', 'unknown')} | chunk {c.get('chunk_index', '?')}]\n{c.get('text', '')}"
        for c in context_chunks
    )

    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "context provided below. If the answer is not in the context, say "
        "'I don't have enough information to answer that based on the available documents.'\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {query}"
    )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    })

    resp = client.invoke_model(
        modelId=LLM_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"]


# ── Main tool function ────────────────────────────────────────────────────────

def rag_search(query: str, top_k: int = RAG_TOP_K) -> dict[str, Any]:
    """
    Answer a question using RAG over the FAISS vector index.

    Args:
        query:  Natural-language question.
        top_k:  Number of chunks to retrieve (default 5).

    Returns:
        {query, answer, retrieved_chunks, sources}
    """
    # Embed query
    q_vector = embed_query(query)

    # Search FAISS
    results = search_faiss(q_vector, top_k=top_k)

    if not results:
        return {
            "query": query,
            "answer": "No documents are indexed yet. Please upload documents first using Tool 1.",
            "retrieved_chunks": 0,
            "sources": [],
        }

    # Generate grounded answer
    answer = _generate_answer(query, results)

    sources = [
        {
            "filename": r.get("filename", "unknown"),
            "s3_key": r.get("s3_key", ""),
            "chunk_index": r.get("chunk_index", 0),
            "score": round(r.get("score", 0.0), 4),
            "excerpt": r.get("text", "")[:300],
        }
        for r in results
    ]

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": len(results),
        "sources": sources,
    }
