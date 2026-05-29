"""
Tool 3 – Document Storage & Relevant Page Search
=================================================
Provides:
  A. search_relevant_pages – k-NN search in FAISS, returns ranked page excerpts.
  B. list_stored_documents – list all unique documents in the FAISS index.
  C. get_document_chunks   – retrieve all chunks for a specific doc_id.

All vector operations use the shared FAISS index managed by utils.py.
"""

from __future__ import annotations

from typing import Any

from utils import (
    embed_query,
    get_all_doc_ids,
    load_faiss_index,
    search_faiss,
)


def search_relevant_pages(
    query: str,
    top_k: int = 5,
    doc_id: str | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """
    Search the FAISS vector index for the most relevant document pages/chunks.

    Args:
        query:    Natural-language search query.
        top_k:    Number of results to return (default 5).
        doc_id:   Optional – filter results to a specific document ID.
        filename: Optional – filter results to a specific filename.

    Returns:
        {query, total_found, results: [{rank, score, doc_id, filename, chunk_index, text_excerpt}]}
    """
    q_vector = embed_query(query)
    # Fetch more than top_k to allow post-filtering
    fetch_k = top_k * 4 if (doc_id or filename) else top_k
    raw_results = search_faiss(q_vector, top_k=fetch_k)

    # Apply optional filters
    if doc_id:
        raw_results = [r for r in raw_results if r.get("doc_id") == doc_id]
    if filename:
        raw_results = [r for r in raw_results if r.get("filename") == filename]

    raw_results = raw_results[:top_k]

    results = [
        {
            "rank": i + 1,
            "score": round(r.get("score", 0.0), 4),
            "doc_id": r.get("doc_id", ""),
            "filename": r.get("filename", ""),
            "s3_key": r.get("s3_key", ""),
            "chunk_index": r.get("chunk_index", 0),
            "text_excerpt": r.get("text", "")[:500],
        }
        for i, r in enumerate(raw_results)
    ]

    return {
        "query": query,
        "total_found": len(results),
        "results": results,
    }


def list_stored_documents() -> dict[str, Any]:
    """
    List all unique documents currently stored in the FAISS index.

    Returns:
        {total_documents, total_vectors, documents: [{doc_id, filename, chunk_count}]}
    """
    index, _ = load_faiss_index()
    total_vectors = index.ntotal if index is not None else 0
    documents = get_all_doc_ids()

    return {
        "total_documents": len(documents),
        "total_vectors": total_vectors,
        "documents": documents,
    }


def get_document_chunks(doc_id: str) -> dict[str, Any]:
    """
    Retrieve all stored chunks for a specific document.

    Args:
        doc_id: The document ID to retrieve chunks for.

    Returns:
        {doc_id, filename, chunks: [{chunk_index, text}]}
    """
    _, metadata = load_faiss_index()
    chunks = [m for m in metadata if m.get("doc_id") == doc_id]
    chunks_sorted = sorted(chunks, key=lambda x: x.get("chunk_index", 0))

    filename = chunks_sorted[0].get("filename", "") if chunks_sorted else ""

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunk_count": len(chunks_sorted),
        "chunks": [
            {"chunk_index": c.get("chunk_index", 0), "text": c.get("text", "")}
            for c in chunks_sorted
        ],
    }
